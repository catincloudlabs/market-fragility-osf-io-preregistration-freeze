"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: OPRA KINEMATICS & FILTERING
=============================================================================
Protocol Reference: 
  - Section 4.2 (Options Kinetic Data)
  - Section 4.2 (Standardized Contract Filter & Expiration Boundary Protocol)

1. PROTOCOL ALIGNMENT (CBBO Extraction):
This DAG successfully extracts the 1-minute Continuous BBO (`cbbo-1m`) aggregates 
required to derive the Net Gamma Potential ($\Psi_{\gamma}$) and the Implied Beta 
($\beta_{IV}$). It perfectly brackets the $t_0$ (09:30) and $t_1$ (10:30) windows.

2. PROTOCOL DELEGATION (Contract Standardization):
By querying the API using `stype_in="parent"`, this script retrieves the ENTIRE 
options chain for the underlying asset. Section 4.2 explicitly mandates the 
exclusion of non-standard OCC deliverables, contracts where the multiplier 
$M \neq 100$, and contracts where $T_{exp} < T_{current}$. 

Because the raw Databento binary (`dbn.zst`) contains all generated contracts, 
the strict application of the Standardized Contract Filter and Expiration Boundary 
Protocol is explicitly delegated to the downstream physics engine (`options_kinematics.py` 
or equivalent). This ensures only valid, standard contracts enter the vectorized 
Black-Scholes pricing kernel.
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

# Standard Airflow Retry (for infrastructure failures)
default_args = {
    "owner": "CatInCloud",
    "retries": 3,
    "retry_delay": pendulum.duration(minutes=2),
}

@dag(
    dag_id="ingest_options_batch",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup=False,
    default_args=default_args,
    tags=["twin", "batch", "opra", "atomic"],
    max_active_tasks=6, 
)
def ingest_options_batch():

    @task
    def ensure_bucket_exists():
        """
        Idempotent check to ensure the simulation lake exists.
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
    def get_ticker_chunks(**context):
        """
        ATOMIC CHUNKING: Returns a list of lists, e.g., [['NVDA'], ['AAPL']].
        Strictly 1 ticker per job to prevent Gateway Timeouts.
        """
        target_tickers = TwinConfig.get_universe()
        
        # Allow manual override via Airflow Configuration JSON
        dag_run = context.get("dag_run")
        if dag_run and dag_run.conf and "tickers" in dag_run.conf:
            target_tickers = [t.strip().upper() for t in dag_run.conf["tickers"].split(",")]
            
        # SANITIZATION FIX: Remove dots for options symbology (e.g., BRK.B -> BRKB)
        sanitized_tickers = [t.replace(".", "") for t in target_tickers]
            
        # Chunking strategy: 1 ticker per job (Atomic)
        return [[ticker] for ticker in sanitized_tickers]

    @task(pool="databento_pool") 
    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def submit_batch_job(tickers: list, **context):
        """
        Submits the job. Retries automatically if the API Gateway flakes out.
        """
        # Jitter to protect the API rate limits
        time.sleep(random.uniform(1.0, 4.0))
        
        client = db.Historical(key=Variable.get("DATABENTO_API_KEY"))
        run_date = context["ds"] # YYYY-MM-DD from Airflow
        
        # Define Time Window (Market Hours)
        et_tz = pendulum.timezone("America/New_York")
        dt_open = pendulum.from_format(f"{run_date} 09:30:00", "YYYY-MM-DD HH:mm:ss", tz=et_tz)
        dt_close = pendulum.from_format(f"{run_date} 16:00:00", "YYYY-MM-DD HH:mm:ss", tz=et_tz)
        
        start_utc = dt_open.in_timezone("UTC").to_iso8601_string()
        end_utc = dt_close.in_timezone("UTC").to_iso8601_string()
        
        # Symbology: 'parent' schema requires .OPT suffix
        formatted_symbols = ",".join([f"{t}.OPT" for t in tickers])

        submission = client.batch.submit_job(
            dataset=TwinConfig.OPTIONS.dataset,
            symbols=formatted_symbols,
            schema=TwinConfig.OPTIONS.schema,
            stype_in=TwinConfig.OPTIONS.stype, 
            start=start_utc, 
            end=end_utc,
            limit=None,
            encoding="dbn",
            compression="zstd"
        )
        return {"job_id": submission["id"], "run_date": run_date, "ticker": tickers[0]}

    @task.sensor(poke_interval=45, timeout=3600, mode="reschedule", pool="databento_pool")
    def wait_for_job_sensor(job_info: dict):
        """
        Polls Databento. Handles 'queued', 'processing', and 'done'.
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
                logger.info(f"✅ Job {job_id} ({ticker}) is DONE. Ready for download.")
                return True 
            elif status in ["received", "queued", "processing", "in_progress"]:
                logger.info(f"⏳ Job {job_id} ({ticker}) is {status.upper()}. Waiting...")
                return False
            elif status in ["red", "failed", "error", "expired", "cancelled"]:
                raise ValueError(f"❌ Job {job_id} ({ticker}) FAILED on server. Status: {status}")
            else:
                logger.warning(f"❓ Job {job_id} ({ticker}) has unknown status: {status}. Waiting...")
                return False
        except Exception as e:
            # Swallow transient network errors during poke to avoid failing the task
            logger.error(f"Error checking job status: {e}")
            return False

    @task(pool="databento_pool")
    def download_and_store(job_info: dict):
        """
        Downloads the specific job artifacts and streams them to MinIO.
        """
        job_id = job_info["job_id"]
        run_date = job_info["run_date"]
        ticker = job_info["ticker"]
        
        logger = logging.getLogger("airflow.task")
        client = db.Historical(key=Variable.get("DATABENTO_API_KEY"))
        
        endpoint = TwinConfig.STORAGE["endpoint_url"].replace("http://", "").replace("https://", "")
        minio_client = Minio(
            endpoint=endpoint,
            access_key=TwinConfig.STORAGE["key"],
            secret_key=TwinConfig.STORAGE["secret"],
            secure=TwinConfig.STORAGE["secure"]
        )
        
        # Local scratch path
        output_dir = Path(f"/tmp/{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"📥 Downloading batch {job_id} for {ticker}...")
            downloaded_files = client.batch.download(
                job_id=job_id, 
                output_dir=output_dir
            )
            
            # 1. Identify ALL dbn.zstd files (Sharding Support)
            data_files = [f for f in downloaded_files if str(f).endswith((".dbn.zst", ".dbn.zstd"))]
            
            if not data_files:
                raise FileNotFoundError(f"❌ No .dbn.zstd files found for job {job_id}")

            # 2. Iterate and Upload ALL shards
            for i, file_path in enumerate(data_files):
                suffix = f"_part{i}" if len(data_files) > 1 else ""
                
                # Naming: raw/opra/batch/YYYY-MM-DD/options_cbbo1m_{TICKER}_{JOBID}.dbn.zstd
                object_name = f"raw/opra/batch/{run_date}/options_cbbo1m_{ticker}_{job_id}{suffix}.dbn.zstd"
                
                minio_client.fput_object(
                    TwinConfig.STORAGE["bucket"], 
                    object_name, 
                    str(file_path)
                )
                logger.info(f"🚀 Successfully uploaded {object_name}")
                
        finally:
            # Cleanup local disk space to prevent Docker container bloat
            if output_dir.exists():
                shutil.rmtree(output_dir)
                logger.info(f"🧹 Cleaned up {output_dir}")

    # --- ORCHESTRATION ---
    bucket_ready = ensure_bucket_exists()
    chunks = get_ticker_chunks()
    
    # 1. Submit jobs
    job_infos = submit_batch_job.expand(tickers=chunks)
    
    # 2. Wait for jobs to finish (Mapped Sensor)
    wait_tasks = wait_for_job_sensor.expand(job_info=job_infos)
    
    # 3. Download & Store (Mapped Task)
    store_tasks = download_and_store.expand(job_info=job_infos)

    bucket_ready >> chunks >> job_infos
    
    # Crucial: The download (store_tasks) must wait for the sensor (wait_tasks) to return True
    job_infos >> wait_tasks >> store_tasks

ingest_options_batch()
