"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: MACRO BOUNDARY ALIGNMENT
=============================================================================
Protocol Reference: 
  - Section 3.3.1 (Merton Model Inputs: Risk-Free Rate)
  - Section 4.2 (Data Acquisition: Macroeconomic boundary conditions)

1. PROTOCOL ALIGNMENT (DGS1 & Forward Fill):
This script successfully extracts the 1-Year US Treasury Constant Maturity Rate 
(DGS1) required to establish the continuous-time risk-free rate ($r_{\text{GS1}}$) 
for the Black-Scholes pricing kernel and Merton Distance-to-Default ($D^*$). 

Per Section 3.3.1, missing data due to SIFMA bond market holidays must be 
forward-filled ($r_t = r_{t-1}$). This script safely captures the null state. 
The mathematical forward-fill operation is explicitly delegated to the 
downstream `state_equations.py` physics engine during the time-series join.

2. PROTOCOL DEVIATION (VIXCLS Extraction):
This PoC script extracts the CBOE Volatility Index (VIXCLS). Note that the VIX 
is strictly an EXOGENOUS macro variable. The theoretical protocol models 
market crashes as ENDOGENOUS phase transitions (Section 1), relying entirely 
on the constituent-specific intrinsic asset volatility ($\sigma_A$) and thermal 
baseline ($\sigma_{\text{base}}$). 

The VIX data is fetched here purely for secondary exploratory visualization in 
the PoC environment and is STRICTLY ISOLATED from the thermodynamic State Equations 
($H_{\text{eff}}$ and $H_c$). It is not a fitted parameter.
=============================================================================
"""

import pendulum
import logging
import requests
import pandas as pd
import shutil
from pathlib import Path
from airflow.decorators import dag, task
from airflow.models import Variable
from minio import Minio
from twin_config import TwinConfig
from tenacity import retry, wait_exponential, stop_after_attempt

default_args = {
    "owner": "CatInCloud",
    "retries": 3,
    "retry_delay": pendulum.duration(minutes=2),
}

@dag(
    dag_id="ingest_boundary_conditions",
    schedule=None,
    # [PROTOCOL ENFORCEMENT] Strict alignment to US Eastern Time 
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup=False,
    default_args=default_args,
    tags=["twin", "batch", "macro", "fred", "empirical"],
)
def ingest_boundary_conditions():

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
    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(3), reraise=True)
    def fetch_fred_data(**context):
        """
        Empirically queries the Federal Reserve API for target macroscopic boundaries.
        """
        run_date = context["ds"] # YYYY-MM-DD
        logger = logging.getLogger("airflow.task")
        
        fred_api_key = Variable.get("FRED_API_KEY")
        
        # Target FRED Series Codes
        # DGS1: 1-Year Treasury Constant Maturity Rate
        # VIXCLS: CBOE Volatility Index (Secondary Context)
        series_codes = ["DGS1", "VIXCLS"]
        
        results = {"date": run_date}
        
        for series in series_codes:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series}&api_key={fred_api_key}&file_type=json"
                f"&observation_start={run_date}&observation_end={run_date}"
            )
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            try:
                # Extract the value, handling FRED's standard '.' for missing/holiday data
                value_str = data["observations"][0]["value"]
                if value_str == ".":
                    results[series] = None
                else:
                    results[series] = float(value_str)
            except (IndexError, KeyError):
                # Occurs if the date is a weekend/holiday and FRED returns an empty observations list
                results[series] = None

        df = pd.DataFrame([results])
        logger.info(f"Empirical Macro Regime for {run_date}:\n{df.to_string()}")

        # --- UPLOAD TO MINIO ---
        endpoint = TwinConfig.STORAGE["endpoint_url"].replace("http://", "").replace("https://", "")
        minio_client = Minio(
            endpoint=endpoint,
            access_key=TwinConfig.STORAGE["key"],
            secret_key=TwinConfig.STORAGE["secret"],
            secure=TwinConfig.STORAGE["secure"]
        )

        output_dir = Path(f"/tmp/macro_{run_date}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            filename = f"macro_indicators_{run_date}.parquet"
            local_path = output_dir / filename
            df.to_parquet(local_path)
            
            object_name = f"raw/macro/{run_date}/{filename}"
            minio_client.fput_object(TwinConfig.STORAGE["bucket"], object_name, str(local_path))
            logger.info(f"🚀 Uploaded empirical FRED data: {object_name}")
            
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)

    ensure_bucket_exists() >> fetch_fred_data()

ingest_boundary_conditions()
