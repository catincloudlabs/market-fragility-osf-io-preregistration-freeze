"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: COMPUTATIONAL INFRASTRUCTURE
=============================================================================
Protocol Reference: 
  - Section 4.1 (Disclosure of Engineering Proof-of-Concept)
  - Section 4.5 (Computational Reproducibility)

1. PROTOCOL ALIGNMENT (Out-of-Core Execution):
Section 4.5 explicitly mandates the use of Polars and DuckDB to ensure 
computational feasibility given the high-dimensional vector operations required 
for the OPRA option lattice integration. This script perfectly satisfies that 
requirement, establishing the in-memory DuckDB instance (via `httpfs`/`aws` extensions) 
and the Polars lazy-scanning architecture required to interface with the MinIO lake.

2. ENGINEERING ARTIFACT (Postgres Sink):
This script introduces a `PostgresConnector`. While PostgreSQL is not explicitly 
named in Section 4.5's list of compute engines, it is utilized here strictly as 
an operational artifact of the PoC environment. It functions purely as a passive 
persistent storage sink for the final calculated diagnostic panel data ($\Xi_{i,t}$) 
and the baseline scoring metrics. It executes NO mathematical transformations or 
state equation logic.
=============================================================================
"""

import polars as pl
import duckdb
import databento as db
from io import BytesIO
from minio import Minio
from dags.twin_config import TwinConfig
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class MinIOConnector:
    def __init__(self):
        # 1. Initialize Standard MinIO Client
        endpoint_clean = TwinConfig.STORAGE["endpoint_url"].replace("http://", "").replace("https://", "")
        
        self.client = Minio(
            endpoint=endpoint_clean,
            access_key=TwinConfig.STORAGE["key"],
            secret_key=TwinConfig.STORAGE["secret"],
            secure=TwinConfig.STORAGE["secure"]
        )
        
        self.bucket = TwinConfig.STORAGE["bucket"]
        
        # 2. Pre-compute Polars Storage Options (for memory-efficient s3fs mapping)
        self.polars_options = TwinConfig.get_polars_storage_options()

    def _normalize_symbol(self, symbol: str) -> str:
        """Enforces CMS standard hyphenation prior to path construction."""
        if not symbol:
            return symbol
        return symbol.replace('.', '-').replace('/', '-')

    def _get_s3_uri(self, object_path):
        """Helper to format s3:// URI for the Polars scanner."""
        return f"s3://{self.bucket}/{object_path}"

    def configure_duckdb(self, con=None):
        """
        Injects MinIO S3 credentials into a DuckDB connection.
        Used for heavy out-of-core SQL aggregations on raw parquet files.
        """
        if con is None:
            con = duckdb.connect()
        
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(TwinConfig.get_duckdb_secret_sql())
        return con

    def scan_parquet(self, object_path):
        """
        Instantiates a Polars LazyFrame. 
        Data is mapped but not loaded into memory until execution (.collect()).
        """
        uri = self._get_s3_uri(object_path)
        try:
            return pl.scan_parquet(
                uri,
                storage_options=self.polars_options
            )
        except Exception as e:
            print(f"❌ Failed to scan {object_path}: {e}")
            return None

    def load_dbn(self, object_path):
        """
        Downloads and decodes a raw Databento .dbn.zstd binary file.
        Uses the Databento SDK to handle symbology mapping automatically,
        then converts to Polars for high-performance physics calculations.
        """
        try:
            print(f"   ⬇️ Streaming DBN bytes: {object_path}")
            response = self.client.get_object(self.bucket, object_path)
            data_bytes = response.read()
            response.close()
            
            # Defensive check: Protect against 0-byte file ingestion
            if len(data_bytes) == 0:
                return None

            # Load into DBNStore
            dbn_store = db.DBNStore.from_bytes(data_bytes)
            
            try:
                # Use to_df() to ensure the 'symbol' column is automatically 
                # mapped from the metadata header by the Databento SDK.
                pandas_df = dbn_store.to_df()
                
                if pandas_df.empty:
                    return None
                
                # Immediately convert to Polars to regain performance
                # We reset the index to ensure 'ts_event' becomes a column
                df = pl.from_pandas(pandas_df.reset_index())
                
                # PROTOCOL ENFORCEMENT: 1e9 Price Scaling (Type-Aware)
                # Databento's SDK automatically converts some schemas to floats but leaves others as ints.
                # We dynamically check the Polars schema and ONLY divide by 1e9 if it is still an integer.
                int_types = [pl.Int64, pl.Int32, pl.UInt64, pl.UInt32]
                price_cols = [
                    c for c in df.columns 
                    if ('price' in c or 'px' in c) and df.schema[c] in int_types
                ]
                
                if price_cols:
                    df = df.with_columns([
                        (pl.col(c) / 1e9).alias(c) for c in price_cols
                    ])
                
            except Exception as e:
                print(f"      ⚠️ Databento materialization failed: {e}")
                return None

            return df
            
        except Exception as e:
            print(f"❌ Failed to load DBN {object_path}: {e}")
            return None

    def _enforce_temporal_geometry(self, df, date_str, snapshot_time, mode="snapshot"):
        """
        PROTOCOL ALIGNMENT: Section 4.2 & Section 4.3
        
        This method physically binds the continuous market data stream into 
        the discrete temporal windows required by the thermodynamic calculations.
        """
        if df is None or snapshot_time is None:
            return df

        # Isolate the nanosecond timestamp column
        ts_col = "ts_event" if "ts_event" in df.columns else "ts_recv" if "ts_recv" in df.columns else None
        if not ts_col:
            return df
            
        try:
            # 1. Establish strict Eastern Time boundaries using Python datetime
            et_zone = ZoneInfo("America/New_York")
            target_dt_et = datetime.strptime(f"{date_str} {snapshot_time}:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=et_zone)
            
            if mode == "snapshot":
                start_et = target_dt_et
                end_et = target_dt_et + timedelta(minutes=1)
            elif mode == "cumulative":
                start_et = datetime.strptime(f"{date_str} 09:30:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=et_zone)
                end_et = target_dt_et 
                # Catch 09:30 zero-window collapse
                if start_et == end_et:
                    end_et = start_et + timedelta(minutes=1)
            else:
                return df
                
            # 2. Convert to UTC
            start_utc = start_et.astimezone(ZoneInfo("UTC"))
            end_utc = end_et.astimezone(ZoneInfo("UTC"))
            
            # 3. Polars String-to-Datetime Cast
            # Format boundaries as strict ISO-8601 strings ending in 'Z' (UTC)
            start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%S.%f000Z")
            end_str = end_utc.strftime("%Y-%m-%dT%H:%M:%S.%f000Z")
            
            # Use native Polars parsing to guarantee type-matching with [ns, UTC]
            start_expr = pl.lit(start_str).str.to_datetime(time_unit="ns", time_zone="UTC")
            end_expr = pl.lit(end_str).str.to_datetime(time_unit="ns", time_zone="UTC")
            
            # 4. Execute Filter
            if mode == "cumulative":
                filtered_df = df.filter(
                    (pl.col(ts_col) >= start_expr) &
                    (pl.col(ts_col) <= end_expr)
                )
            else:
                filtered_df = df.filter(
                    (pl.col(ts_col) >= start_expr) &
                    (pl.col(ts_col) < end_expr)
                )
            
            return filtered_df
            
        except Exception as e:
            print(f"      ⚠️ Temporal Geometry filter failed: {e}")
            return df

    def get_equity_snapshot(self, symbol, date, snapshot_time=None):
        """
        Retrieves Limit Order Book (L2/MBP-10) Equity Data.
        Applies Protocol Section 4.2 bounds if snapshot_time is provided.
        """
        symbol = self._normalize_symbol(symbol)
        path = f"raw/equity/{symbol}/{date}/"
        try:
            objects = self.client.list_objects(self.bucket, prefix=path, recursive=True)
            files = sorted([obj.object_name for obj in objects])
            
            candidates = [f for f in files if "ohlcv" not in f]
            if not candidates:
                candidates = files 
            
            if not candidates:
                raise FileNotFoundError(f"No equity data found for {symbol} on {date}")

            target_file = candidates[0]
            
            if target_file.endswith(".parquet"):
                df = self.scan_parquet(target_file)
            elif target_file.endswith(".dbn.zstd") or target_file.endswith(".dbn"):
                df = self.load_dbn(target_file)
            else:
                raise ValueError(f"Unknown format: {target_file}")
                
            return self._enforce_temporal_geometry(df, date, snapshot_time, mode="snapshot")
                
        except Exception as e:
            print(f"❌ Error listing equity objects: {e}")
            return None

    def get_ohlcv_snapshot(self, symbol, date, snapshot_time=None):
        """
        Retrieves OHLCV price action data. 
        Isolates specific 1-min interval if requested.
        """
        symbol = self._normalize_symbol(symbol)
        path = f"raw/equity/{symbol}/{date}/"
        try:
            objects = self.client.list_objects(self.bucket, prefix=path, recursive=True)
            files = [obj.object_name for obj in objects if "ohlcv" in obj.object_name]
            
            if not files:
                print(f"⚠️ No OHLCV data found in {path}")
                return None
            
            df = self.load_dbn(files[0])
            return self._enforce_temporal_geometry(df, date, snapshot_time, mode="snapshot")
            
        except Exception as e:
            print(f"❌ Error listing OHLCV objects: {e}")
            return None

    def get_ohlcv_history(self, symbol):
        """
        Retrieves the 252-day trailing historical baseline context.
        Required for calculating macroscopic velocity (Minsky Beta_M).
        """
        symbol = self._normalize_symbol(symbol)
        path = f"raw/equity/{symbol}/history/"
        try:
            return self.scan_parquet(path + "*.parquet")
        except Exception as e:
            print(f"⚠️ OHLCV History not found for {symbol}: {e}")
            return None

    def get_option_surface(self, symbol, date, snapshot_time=None, time_mode="cumulative"):
        """
        Retrieves raw Option Chain (OPRA) data. 
        By default, time_mode="cumulative" aggregates total contract volume 
        from the 09:30 open to proxy the Active Hedging Mass.
        """
        symbol = self._normalize_symbol(symbol)
        path = f"raw/opra/batch/{date}/"
        try:
            objects = self.client.list_objects(self.bucket, prefix=path, recursive=True)
            files = sorted([obj.object_name for obj in objects])
            
            # Exact match filtering to prevent "A" from returning "AAPL" data
            target_files = [f for f in files if f"_{symbol}_" in f]
            
            if not target_files:
                return None
                
            target_file = target_files[0]
            
            if target_file.endswith(".parquet"):
                df = self.scan_parquet(target_file)
            elif target_file.endswith(".dbn.zstd") or target_file.endswith(".dbn"):
                df = self.load_dbn(target_file)
            else:
                 raise ValueError(f"Unknown format: {target_file}")
                 
            return self._enforce_temporal_geometry(df, date, snapshot_time, mode=time_mode)
                 
        except Exception as e:
            print(f"❌ Error finding option surface: {e}")
            return None

    def save_parquet(self, df: pl.DataFrame, object_path: str):
        """Writes a computed Polars DataFrame back to the MinIO object store."""
        try:
            data = BytesIO()
            df.write_parquet(data)
            data.seek(0)
            
            self.client.put_object(
                self.bucket,
                object_path,
                data,
                len(data.getbuffer()),
                content_type="application/x-parquet"
            )
            print(f"   💾 Saved: {object_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save {object_path}: {e}")
            return False
