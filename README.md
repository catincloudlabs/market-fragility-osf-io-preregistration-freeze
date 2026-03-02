# CatInCloud Labs: Protocol #1

## The Thermodynamics of Market Fragility

**Welcome to CatInCloud Labs.** We are an independent Product R&D lab bridging the gap between theoretical physics, structural thermodynamics, and quantitative data engineering.

This repository contains the foundational Proof-of-Concept (PoC) computational engine for our first hypothesis: **Protocol #1**.

### The Hypothesis

We believe the era of the proprietary, black-box quantitative risk model is fundamentally limited by its reliance on continuous-time, infinite-liquidity assumptions. Standard financial models treat market crashes as exogenous tail events (unpredictable >3-sigma outliers).

We propose a structural alternative: **Extreme market events are endogenous phase transitions resulting from the accumulation of stress in a finite-rate open system.** This repository contains the institutional-grade physics engine built to execute that theory. It maps high-frequency market microstructure (nanosecond-timestamped Level 2 Order Books and OPRA flow) to a 4-dimensional state-space manifold, calculating:

* **The Minsky Betas:** A thermodynamic tetrad governing macroscopic kinetic, localized potential, and implied market temperatures.
* **Effective Viscosity ($\eta_{\mathcal{O}}$):** The kinetic friction of the deterministic dealer options field.
* **The Criticality Index ($\Xi$):** A continuous diagnostic of structural metastability and phase failure (Sublimation vs. Viscous Relaxation).

Here is the math. Here is the data pipeline. Here is the codebase. Tear it apart.

---

## OSF Preregistration & PoC Disclosure

This codebase is intrinsically linked to our formal theoretical paper and preregistration on the Open Science Framework (OSF).

**CRITICAL SCOPE BOUNDARY:** To maintain absolute epistemological integrity and prevent look-ahead bias, the formal Validation Cohort (2019–2025) remains strictly unobserved by this codebase.

This repository represents the **Engineering Proof-of-Concept (PoC)**. It is localized to a strict out-of-sample micro-window (Jan 5 – Feb 20, 2026). It utilizes a locally emulated MinIO data lake, a synthetic 252-day macroscopic heat bath (via Monte Carlo GBM), and static cohort files to prove the *computational architecture* and *dimensional stability* of the state equations without exposing proprietary cloud infrastructure or violating data licensing constraints.

---

## 📁 Architecture

This project is orchestrated using an isolated, Dockerized Apache Airflow environment, utilizing Polars and DuckDB for high-performance out-of-core vectorized execution.

### Layer 1: Data Acquisition (`airflow-docker-minio-backend/dags/`)

Automates the asynchronous ingestion of raw telemetry into the MinIO lake.

* `ingest_equity_batch.py`: Extracts L2 MBP-10 structural data.
* `ingest_equity_ohlcv_batch.py`: Extracts OHLCV structural data.
* `ingest_options_batch.py`: Extracts high-resolution OPRA liquidity.
* `ingest_boundary_conditions.py` & `ingest_fundamentals_mock.py`: Establishes point-in-time boundary conditions.

### Layer 2: The Physics Engine (`twin_stack_simulation/`)

Executes the discrete-time geometry and thermodynamic state equations.

* `connectors.py`: Handles I/O streaming and lazy-evaluation of parquet/dbn binaries, enforcing a $1e9$ scalar defense for fixed-point prices.
* `kinematics.py`: Vectorized Black-Scholes continuous-time solver (strictly enforcing $q=0$ for Maximal Fragility).
* `thermodynamics.py`: Calculates orthogonal inverse temperatures ($\beta$) and Newtonian mass limits.
* `topology.py`: Derives Liquidity Capacity ($\Psi_{liq}$), Net Gamma Potential ($\Psi_{\gamma}$), and Effective Viscosity.
* `solvency.py`: Computes the Merton Distance-to-Default barrier utilizing a Maximum Entropy (MaxEnt) logistic state occupancy.
* `stress.py`: Aggregates the Hamiltonian ($H_{eff}$) by integrating kinetic flux, potential strain, and kinetic dissipation.
* `scoring.py`: Evaluates predictive validity (AUPRC) against classical baselines, featuring automated RiskMetrics EWMA fallbacks for non-stationary GARCH convergence failures.

---

## 🚀 Layer 1 - Local Deployment Quick Start

**Prerequisites:**

1. **Astronomer CLI:** For managing the Airflow Docker containers.

2. **API Credentials:** Active keys for Databento (Level 2/OPRA flow) and FRED (Macro rates).

**Deployment Steps:**

1. Clone this repository.

2. Duplicate `.env.example`, rename it to `.env`, and securely inject your API keys.

3. Start the Airflow orchestration engine and local MinIO Data Lake by running:

    ```bash
    astro dev start
    ```

4. Once initialized, navigate to `http://localhost:8080/` to access the Airflow UI and trigger the DAGs.

---

## ⚛️ Layer 2 - Physics Engine Script Execution

**Prerequisites:**

1. Python 3.12+ (The PoC was successfully executed and validated on Python 3.14.0).

2. Completed Layer 1 Ingestion: The raw structural data must already reside in your local MinIO bucket.

**Deployment Steps:**

1. Initialize Environment: Activate your local virtual environment and install the strictly pinned dependencies to guarantee mathematical reproducibility:

    ```bash
    py -m pip install -r requirements.txt
    ```

2. Execute the Pipeline Sequentially: From the root directory of the repository, execute the physics engine modules in the following chronological order. This guarantees the macroscopic boundaries are established before the continuous-time Greeks are solved, which in turn must precede the final Hamiltonian stress aggregations.

    Step 1: Establish the Macroscopic Thermal Bath

    ```bash
    py generate_synthetic_history.py
    ```

    Step 2: Clean and Normalize the OPRA Derivative Lattice

    ```bash
    py transcode_options.py
    ```

    Step 3: Solve the Continuous-Time Kinematics (Greeks & IV)

    ```bash
    py compute_surface.py
    ```

    Step 4: Execute the Discrete Diagnostic Solvers

    ```bash
    py -m physics_engine.thermodynamics
    py -m physics_engine.topology
    py -m physics_engine.solvency
    ```

    Step 5: Aggregate the Systemic Stress Load (The Hamiltonian)

    ```bash
    py -m physics_engine.stress
    ```

    Step 6: Run Simulation

    ```bash
    py run_batch_simulation.py
    ```

    Step 7: Evaluate Statistical Predictive Validity (AUPRC vs Baselines)

    ```bash
    py scoring.py
    ```

---

## License & Disclaimer

* **License:** This project is licensed under the MIT License. See the `LICENSE` file for details.
* **Disclaimer:** This codebase and the accompanying Protocol #1 paper are provided strictly for academic, research, and R&D purposes. CatInCloud Labs is a research entity, not a financial advisor. This software does not constitute financial advice, investment recommendations, or an offer to sell securities. The user assumes all liability for any financial losses or damages incurred through the use of this theoretical engine in live markets.

**Contact:** To report a bug, suggest a phase-state update, or inquire about enterprise implementations, please open an issue in this repository.
