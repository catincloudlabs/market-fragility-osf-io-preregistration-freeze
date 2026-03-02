"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: MICROSTRUCTURE EXTRACTION
=============================================================================
Protocol Reference: 
  - Section 4.2 (Data Acquisition: Visible Market Data & Volume Aggregation)
  - Section 4.3 (Simulation Protocol: Dual-Window Sampling)

1. PROTOCOL ALIGNMENT (Temporal Isolation):
This DAG strictly enforces the temporal boundaries defined in Section 4.2. 
By explicitly bounding the extraction to 09:30:00 - 10:31:00 ET, it isolates 
the required $t_0$ (09:30) and $t_1$ (10:30) snapshot windows, minimizing 
storage overhead while guaranteeing that no look-ahead data ($t > 10:31$) 
enters the raw data lake.

2. PROTOCOL DELEGATION (Fixed-Point Scaling):
Per Section 4.2, raw Databento prices are scaled by $1e9$ to prevent 
floating-point precision loss. This DAG extracts and stores the data in 
its native, unscaled binary format (`dbn.zst`). 

The mandatory scalar reversal (dividing prices by $1e9$ to restore decimal 
floats prior to continuous-time calculations) is explicitly delegated to the 
downstream `state_equations.py` physics engine. This ensures the raw data 
lake remains functionally immutable.
=============================================================================
"""

import pendulum
import logging
import time
import os
import random
import shutil
from pathlib import Path
from airflow.decorators import dag, task
from airflow.models import Variable
from minio import Minio
import databento as db
from twin_config import TwinConfig
from tenacity import retry, wait_exponential, stop_after_attempt

default_args = {
    "owner": "CatInCloud",
    "retries": 5,
    "retry_delay": pendulum.duration(minutes=1),
}

@dag(
    dag_id="ingest_equity_batch",
    schedule=None,
    # [PROTOCOL ENFORCEMENT] Strict alignment to US Eastern Time to prevent UTC bleed
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup=False,
    default_args=default_args,
    tags=["twin", "batch", "equity", "mbp-10"],
    max_active_tasks=6, 
)
def ingest_equity_batch():
    
    @task
    def ensure_bucket_exists():
        """
        Verify the existence of the raw data lake (MinIO).
        Robustly handles Docker networking (http vs https).
        """
        endpoint = TwinConfig.STORAGE["endpoint_url"].replace("http://", "").replace("https://", "")
        
        client = Minio(
            endpoint=endpoint,
            access_key=TwinConfig.STORAGE["key"],
            secret_key=TwinConfig.STORAGE["secret"],
            secure=TwinConfig.STORAGE["secure"]
        )
        if not client.bucket_exists(TwinConfig.STORAGE["bucket"]):
            client.make_bucket(TwinConfig.STORAGE["bucket"])

    @task
    def get_target_tickers(**context):
        """
        Ticker Selection:
        Uses the centralized TwinConfig universe but allows manual override.
        """
        # Default to the centralized Universe file
        target_tickers = TwinConfig.get_universe()

        # Allow manual override via Airflow Configuration JSON
        dag_run = context.get("dag_run")
        if dag_run and dag_run.conf and "tickers" in dag_run.conf:
            target_tickers = [t.strip().upper() for t in dag_run.conf["tickers"].split(",")]
            
        return target_tickers

    @task(pool="databento_pool") 
    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def submit_equity_job(ticker: str, **context):
        """
        Submits an MBP-10 request. Retries on Gateway Timeouts.
        """
        # Jitter to protect the API
        time.sleep(random.uniform(1, 3)) 
        
        client = db.Historical(key=Variable.get("DATABENTO_API_KEY"))
        run_date = context["ds"]

        et_tz = pendulum.timezone("America/New_York")
        dt_open = pendulum.from_format(f"{run_date} 09:30:00", "YYYY-MM-DD HH:mm:ss", tz=et_tz)
        dt_close = pendulum.from_format(f"{run_date} 10:31:00", "YYYY-MM-DD HH:mm:ss", tz=et_tz)
        
        start_utc = dt_open.in_timezone("UTC").to_iso8601_string()
        end_utc = dt_close.in_timezone("UTC").to_iso8601_string()

        submission = client.batch.submit_job(
            dataset=TwinConfig.EQUITY.dataset,
            symbols=ticker,
            schema=TwinConfig.EQUITY.schema, 
            stype_in=TwinConfig.EQUITY.stype, 
            start=start_utc, 
            end=end_utc,
            limit=None,
            encoding="dbn",
            compression="zstd"
        )
        # Pass all info needed for the next steps
        return {"job_id": submission["id"], "ticker": ticker, "run_date": run_date}

    @task.sensor(poke_interval=45, timeout=3600, mode="reschedule", pool="databento_pool")
    def wait_for_job_sensor(job_info: dict):
        """
        Polls Databento. Hardened to handle all status codes.
        """
        job_id = job_info["job_id"]
        ticker = job_info["ticker"]
        client = db.Historical(key=Variable.get("DATABENTO_API_KEY"))
        logger = logging.getLogger("airflow.task")
        
        try:
            all_jobs = client.batch.list_jobs()
            job_status = next((job for job in all_jobs if job.get("id") == job_id), None)
            
            if not job_status:
                logger.warning(f"⚠️ Job {job_id} ({ticker}) not found yet. Retrying...")
                return False

            status = job_status.get("state", job_status.get("status", "")).lower()
            
            if status == "done": 
                logger.info(f"✅ Job {job_id} ({ticker}) is DONE.")
                return True 
            elif status in ["received", "queued", "processing", "in_progress"]:
                logger.info(f"⏳ Job {job_id} ({ticker}) is {status.upper()}. Waiting...")
                return False
            elif status in ["red", "failed", "error", "expired", "cancelled"]:
                raise ValueError(f"❌ Job {job_id} ({ticker}) FAILED on server. Status: {status}")
            else:
                return False 
        except Exception as e:
            logger.error(f"Error checking status for {ticker}: {e}")
            return False

    @task(pool="databento_pool")
    def download_and_store_equity(job_info: dict):
        """
        Downloads data and streams to MinIO.
        Includes cleanup logic to prevent Docker container bloat.
        """
        logger = logging.getLogger("airflow.task")
        client = db.Historical(key=Variable.get("DATABENTO_API_KEY"))
        
        job_id = job_info["job_id"]
        ticker = job_info["ticker"]
        run_date = job_info["run_date"]
        
        # Parse endpoint from TwinConfig
        endpoint = TwinConfig.STORAGE["endpoint_url"].replace("http://", "").replace("https://", "")
        minio_client = Minio(
            endpoint=endpoint,
            access_key=TwinConfig.STORAGE["key"],
            secret_key=TwinConfig.STORAGE["secret"],
            secure=TwinConfig.STORAGE["secure"]
        )
        
        output_dir = Path(f"/tmp/{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
            
        try:
            logger.info(f"📥 Downloading job {job_id}...")
            # Capture the specific files downloaded
            downloaded_files = client.batch.download(
                job_id=job_id, 
                output_dir=output_dir
            )
            
            target_file = None
            for file_path in downloaded_files:
                if str(file_path).endswith((".dbn.zst", ".dbn.zstd")):
                    target_file = file_path
                    break
                
            if target_file:
                # Path: raw/equity/{ticker}/{date}/{ticker}_mbp10.dbn.zstd
                object_name = f"raw/equity/{ticker}/{run_date}/{ticker}_mbp10.dbn.zstd"
                minio_client.fput_object(TwinConfig.STORAGE["bucket"], object_name, str(target_file))
                logger.info(f"🚀 Successfully uploaded {object_name}")
            else:
                raise FileNotFoundError(f"❌ No .dbn.zstd found for job {job_id}")

        finally:
            # CLEANUP: Crucial for Docker environments
            if output_dir.exists():
                shutil.rmtree(output_dir)
                logger.info(f"🧹 Cleaned up {output_dir}")

    # --- ORCHESTRATION ---
    bucket_ready = ensure_bucket_exists()
    tickers = get_target_tickers()
    
    # 1. Submit all jobs (Atomic: 1 ticker per task)
    job_infos = submit_equity_job.expand(ticker=tickers)
    
    # 2. Wait for jobs (Async Sensor)
    wait_tasks = wait_for_job_sensor.expand(job_info=job_infos)
    
    # 3. Download/Store (Triggered by successful wait)
    store_tasks = download_and_store_equity.expand(job_info=job_infos)

    bucket_ready >> tickers >> job_infos >> wait_tasks >> store_tasks

ingest_equity_batch()
