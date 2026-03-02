"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: FUNDAMENTALS & LOOK-AHEAD PREVENTION
=============================================================================
Protocol Reference: 
  - Section 3.2.5 (Solvency Boundary: Merton Distance-to-Default)
  - Section 4.2 (Fundamental Data & Look-Ahead Bias Prevention)

1. DATA LICENSING & REPOSITORY OMISSION:
To strictly comply with the proprietary data redistribution restrictions of 
Sharadar / Nasdaq Data Link, the raw fundamental data underlying the Merton 
Solvency Boundary ($\Psi_{\text{val}}$) cannot be shared publicly. 
Consequently, the static data file (`2026-SHARADAR-SF1-1.csv`) utilized 
by this script is deliberately excluded from the OSF repository.

2. PROTOCOL DEVIATION (Static PoC Ingestion):
For the out-of-sample Proof-of-Concept (PoC), this script operates as a "mock" 
ingestion DAG. Rather than querying the Sharadar API dynamically, it reads the 
locally staged CSV to populate the MinIO data lake, ensuring the codebase is 
computationally reproducible. 

3. PROTOCOL DELEGATION (Look-Ahead Bias Prevention):
CRITICAL: While this DAG loads the entire historical CSV into the data lake 
at once, the downstream physics engine (`solvency.py`) strictly enforces a 
point-in-time lookback filter (`eval_date <= target_date`). This mathematically 
guarantees that the Distance-to-Default ($D^*$) values update ONLY on the 
strictly reported SEC filing dates, satisfying Section 4.2.
=============================================================================
"""

import pendulum
import logging
import polars as pl
import os
import shutil
from pathlib import Path
from airflow.decorators import dag, task
from minio import Minio
from twin_config import TwinConfig

default_args = {
    "owner": "CatInCloud",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}

@dag(
    dag_id="ingest_fundamentals_mock",
    schedule=None,
    # [PROTOCOL ENFORCEMENT] Strict alignment to US Eastern Time 
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup=False,
    default_args=default_args,
    tags=["twin", "batch", "fundamentals", "static", "empirical"],
)
def ingest_fundamentals_mock():

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
    def process_fundamentals():
        """
        Loads the FULL mock dataset and uploads it to MinIO.
        No date filtering applied here; filtering occurs in solvency.py.
        """
        logger = logging.getLogger("airflow.task")
        
        # Check if source file exists in the container
        if not os.path.exists(TwinConfig.FUNDAMENTALS_PATH):
            raise FileNotFoundError(
                f"❌ Source file not found at {TwinConfig.FUNDAMENTALS_PATH}. "
                "Ensure '2026-SHARADAR-SF1-1.csv' is deployed to the Docker image."
            )

        # Load Full Dataset (No Filter)
        logger.info(f"📂 Loading full dataset from {TwinConfig.FUNDAMENTALS_PATH}...")
        df = pl.read_csv(TwinConfig.FUNDAMENTALS_PATH)
        
        if df.is_empty():
            logger.warning("⚠️ The source CSV is empty. Nothing to upload.")
            return

        count = len(df)
        logger.info(f"✅ Loaded {count} records (Full Universe).")

        # --- UPLOAD ---
        endpoint = TwinConfig.STORAGE["endpoint_url"].replace("http://", "").replace("https://", "")
        minio_client = Minio(
            endpoint=endpoint,
            access_key=TwinConfig.STORAGE["key"],
            secret_key=TwinConfig.STORAGE["secret"],
            secure=TwinConfig.STORAGE["secure"]
        )

        # Temp location
        output_dir = Path("/tmp/fund_static")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # We name it 'full' to indicate it contains all history
            filename = "sharadar_sf1_full.parquet"
            local_path = output_dir / filename
            
            # Write Parquet
            df.write_parquet(local_path)
            
            # Path: raw/fundamentals/static/sharadar_sf1_full.parquet
            object_name = f"raw/fundamentals/static/{filename}"
            
            minio_client.fput_object(TwinConfig.STORAGE["bucket"], object_name, str(local_path))
            logger.info(f"🚀 Successfully uploaded full reference dataset to: {object_name}")
            
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)

    ensure_bucket_exists() >> process_fundamentals()

ingest_fundamentals_mock()
