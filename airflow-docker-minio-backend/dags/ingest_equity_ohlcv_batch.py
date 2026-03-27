"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: OHLCV & LOOK-AHEAD PREVENTION
=============================================================================
Protocol Reference: 
  - Section 4.2 (OHLCV Aggregation Requirement)
  - Section 4.2 (Volume Aggregation Protocol / Look-Ahead Prevention)

1. PROTOCOL ALIGNMENT (Aggregation Requirement):
This DAG successfully extracts the 1-minute OHLCV aggregates required to 
calculate the Kinetic Flux ($Z_{\Phi}$) and perform Volume-Weighted Average Price 
(VWAP) smoothing, explicitly capturing the physical volatility range (High/Low) 
rather than relying on point-in-time ticks.

2. PROTOCOL DELEGATION (Look-Ahead Volume Truncation):
To ensure the 10:30 ET ($t_1$) snapshot is fully captured, this extraction 
window runs until 10:31:00 ET. However, Section 4.2 mandates a strict 
inclusive boundary of 10:30:00.000 ET for all flow variables (Volume) to 
prevent look-ahead bias. 

The truncation of the remaining 59.999 seconds of the 10:30 minute bar is 
explicitly delegated to the downstream `state_equations.py` engine. This DAG 
safely over-fetches the boundary minute by design to ensure the physics engine 
has the complete temporal state required to enforce the strict protocol limit.
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
    "retries": 3,
    "retry_delay": pendulum.duration(minutes=1),
}

@dag(
    dag_id="ingest_equity_ohlcv_batch",
    schedule=None,
    # [PROTOCOL ENFORCEMENT] Strict alignment to US Eastern Time to prevent UTC bleed
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup=False,
    default_args=default_args,
    tags=["twin", "batch", "equity", "ohlcv-1m"],
    max_active_tasks=6, 
)
def ingest_equity_ohlcv_batch():
    
    @task
    def ensure_bucket_exists():
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
        target_tickers = TwinConfig.get_universe()
        dag_run = context.get("dag_run")
        if dag_run and dag_run.conf and "tickers" in dag_run.conf:
            target_tickers = [t.strip().upper() for t in dag_run.conf["tickers"].split(",")]
        return target_tickers

    @task(pool="databento_pool") 
    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5), reraise=True)
    def submit_ohlcv_job(ticker: str, **context):
        """
        Submits an OHLCV-1m request using TwinConfig.EQUITY_OHLCV settings.
        """
        # Lower jitter (OHLCV is cheaper/faster to request)
        time.sleep(random.uniform(0.5, 2.0)) 
        
        client = db.Historical(key=Variable.get("DATABENTO_API_KEY"))
        run_date = context["ds"]

        et_tz = pendulum.timezone("America/New_York")
        dt_open = pendulum.from_format(f"{run_date} 09:30:00", "YYYY-MM-DD HH:mm:ss", tz=et_tz)
        dt_close = pendulum.from_format(f"{run_date} 10:31:00", "YYYY-MM-DD HH:mm:ss", tz=et_tz)
        
        start_utc = dt_open.in_timezone("UTC").to_iso8601_string()
        end_utc = dt_close.in_timezone("UTC").to_iso8601_string()

        # Uses EQUITY_OHLCV config
        submission = client.batch.submit_job(
            dataset=TwinConfig.EQUITY_OHLCV.dataset,
            symbols=ticker,
            schema=TwinConfig.EQUITY_OHLCV.schema, 
            stype_in=TwinConfig.EQUITY_OHLCV.stype, 
            start=start_utc, 
            end=end_utc,
            limit=None,
            encoding="dbn",
            compression="zstd"
        )
        return {"job_id": submission["id"], "ticker": ticker, "run_date": run_date}

    @task.sensor(poke_interval=30, timeout=1800, mode="reschedule", pool="databento_pool")
    def wait_for_job_sensor(job_info: dict):
        job_id = job_info["job_id"]
        ticker = job_info["ticker"]
        client = db.Historical(key=Variable.get("DATABENTO_API_KEY"))
        logger = logging.getLogger("airflow.task")
        
        try:
            all_jobs = client.batch.list_jobs()
            job_status = next((job for job in all_jobs if job.get("id") == job_id), None)
            
            if not job_status:
                logger.warning(f"⚠️ Job {job_id} ({ticker}) not found yet.")
                return False

            status = job_status.get("state", job_status.get("status", "")).lower()
            
            if status == "done": 
                logger.info(f"✅ Job {job_id} ({ticker}) is DONE.")
                return True 
            elif status in ["received", "queued", "processing", "in_progress"]:
                logger.info(f"⏳ Job {job_id} ({ticker}) is {status.upper()}.")
                return False
            elif status in ["red", "failed", "error", "expired", "cancelled"]:
                raise ValueError(f"❌ Job {job_id} ({ticker}) FAILED: {status}")
            else:
                return False 
        except Exception as e:
            logger.error(f"Error checking status: {e}")
            return False

    @task(pool="databento_pool")
    def download_and_store_ohlcv(job_info: dict):
        logger = logging.getLogger("airflow.task")
        client = db.Historical(key=Variable.get("DATABENTO_API_KEY"))
        
        job_id = job_info["job_id"]
        ticker = job_info["ticker"]
        run_date = job_info["run_date"]
        
        endpoint = TwinConfig.STORAGE["endpoint_url"].replace("http://", "").replace("https://", "")
        minio_client = Minio(
            endpoint=endpoint,
            access_key=TwinConfig.STORAGE["key"],
            secret_key=TwinConfig.STORAGE["secret"],
            secure=TwinConfig.STORAGE["secure"]
        )
        
        output_dir = Path(f"/tmp/{job_id}_ohlcv")
        output_dir.mkdir(parents=True, exist_ok=True)
            
        try:
            logger.info(f"📥 Downloading OHLCV job {job_id}...")
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
                object_name = f"raw/equity/{ticker}/{run_date}/{ticker}_ohlcv1m.dbn.zstd"
                minio_client.fput_object(TwinConfig.STORAGE["bucket"], object_name, str(target_file))
                logger.info(f"🚀 Uploaded {object_name}")
            else:
                raise FileNotFoundError(f"❌ No .dbn.zstd found for job {job_id}")

        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)

    # --- ORCHESTRATION ---
    bucket_ready = ensure_bucket_exists()
    tickers = get_target_tickers()
    job_infos = submit_ohlcv_job.expand(ticker=tickers)
    wait_tasks = wait_for_job_sensor.expand(job_info=job_infos)
    store_tasks = download_and_store_ohlcv.expand(job_info=job_infos)

    bucket_ready >> tickers >> job_infos >> wait_tasks >> store_tasks

ingest_equity_ohlcv_batch()
