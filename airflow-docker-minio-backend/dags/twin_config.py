"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: GLOBAL CONFIGURATION & INFRASTRUCTURE
=============================================================================
Protocol Reference: 
  - Section 4.1 (Disclosure of Engineering Proof-of-Concept)
  - Section 4.2 (Data Acquisition: Symbology & Venue Consistency)
  - Section 4.5 (Computational Reproducibility)

1. PROTOCOL ALIGNMENT (Venue Consistency):
Section 4.2 explicitly mandates that all Level 2 microstructure data must be 
sourced exclusively from the Nasdaq matching engine to prevent unobserved 
fragmentation bias. The `DataSourceConfig` classes below strictly enforce 
this by pinning `dataset="XNAS.ITCH"`. 

2. PROTOCOL ALIGNMENT (Cohort Integrity):
The `get_universe()` method strictly enforces the presence of the static 
`universe_2026.csv` file, ensuring the simulation cannot proceed if the 
immutable cohort file is missing. 

3. ENGINEERING ARTIFACT (Storage Emulation):
To ensure the standalone codebase is reproducible without exposing proprietary 
cloud buckets or credentials, this configuration implements a local MinIO 
container (`host.docker.internal`). This serves as a functionally equivalent 
S3 data lake, allowing the exact Polars/DuckDB out-of-core scanning logic 
(mandated in Section 4.5) to be executed and tested locally.
=============================================================================
"""

import os
import polars as pl
from dataclasses import dataclass

@dataclass
class DataSourceConfig:
    dataset: str
    schema: str
    stype: str

class TwinConfig:
    """
    Central configuration for the 2026 Twin Simulation.
    """
    
    # --- PATHS & ENVIRONMENT ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UNIVERSE_PATH = os.path.join(BASE_DIR, "universe_2026.csv")
    FUNDAMENTALS_PATH = os.path.join(BASE_DIR, "2026-SHARADAR-SF1-1.csv")
    
    # --- DATASOURCES ---
    # [PROTOCOL ENFORCEMENT] Strict alignment to XNAS for Venue Consistency
    EQUITY = DataSourceConfig(dataset="XNAS.ITCH", schema="mbp-10", stype="raw_symbol")
    EQUITY_OHLCV = DataSourceConfig(dataset="XNAS.ITCH", schema="ohlcv-1m", stype="raw_symbol")
    OPTIONS = DataSourceConfig(dataset="OPRA.PILLAR", schema="cbbo-1m", stype="parent")

    # --- DYNAMIC STORAGE CONFIGURATION ---
    # Detect if running inside Airflow container (Astro sets AIRFLOW_HOME)
    _is_in_docker = os.getenv("AIRFLOW_HOME") is not None
    
    # If in Docker, talk to the service 'minio'. If local, talk to 'localhost'.
    _minio_host = "host.docker.internal" if _is_in_docker else "localhost"
    
    # Local reproducible S3 credentials (safe to commit for PoC)
    STORAGE = {
        "bucket": "twin-lake-raw",
        "endpoint_url": f"http://{_minio_host}:9000", 
        "key": "admin",                      
        "secret": "password123",             
        "secure": False 
    }
    
    BATCH = {
        "packaging": "dbn",
        "compression": "zstd" 
    }

    @classmethod
    def get_polars_storage_options(cls):
        """
        Returns the dictionary required by Polars (via s3fs) to talk to MinIO.
        """
        return {
            "aws_endpoint_url": cls.STORAGE["endpoint_url"],
            "aws_access_key_id": cls.STORAGE["key"],
            "aws_secret_access_key": cls.STORAGE["secret"],
            "aws_region": "us-east-1",  # MinIO ignores this, but s3fs requires it
            "aws_allow_http": "true",   # Crucial for local MinIO (no SSL)
            "aws_s3_allow_unsafe_rename": "true"
        }

    @classmethod
    def get_duckdb_secret_sql(cls):
        """
        Returns the SQL query to configure DuckDB for MinIO.
        """
        # DuckDB prefers the endpoint without the (http://)
        endpoint = cls.STORAGE["endpoint_url"].replace("http://", "").replace("https://", "")
        return f"""
            CREATE OR REPLACE SECRET minio_secret (
                TYPE S3,
                KEY_ID '{cls.STORAGE["key"]}',
                SECRET '{cls.STORAGE["secret"]}',
                REGION 'us-east-1',
                ENDPOINT '{endpoint}',
                USE_SSL false,
                URL_STYLE 'path'
            );
        """

    @classmethod
    def get_universe(cls):
        """
        Retrieves the asset universe. 
        Strictly enforces the presence of the universe file to maintain cohort integrity.
        """
        if not os.path.exists(cls.UNIVERSE_PATH):
            raise FileNotFoundError(
                f"CRITICAL: Universe file missing at {cls.UNIVERSE_PATH}. "
                "Cohort selection cannot proceed without violating Protocol Sec 4.2."
            )
            
        try:
            df = pl.read_csv(cls.UNIVERSE_PATH)
            df = df.rename({col: col.lower() for col in df.columns})
            
            for valid_col in ["symbol", "ticker", "root_symbol"]:
                if valid_col in df.columns:
                    return df[valid_col].str.strip_chars().unique().to_list()
            
            raise ValueError("Universe CSV is missing a valid symbol column.")
            
        except Exception as e:
            raise RuntimeError(f"Error reading universe: {e}")
