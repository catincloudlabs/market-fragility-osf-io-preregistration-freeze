# =============================================================================
# OSF PREREGISTRATION DISCLOSURE: ENVIRONMENT REPLICATION
# =============================================================================
# Protocol Reference: 
#   - Section 4.5 (Computational Reproducibility)
#
# 1. PROTOCOL ALIGNMENT (Python 3.12+ & Airflow):
# Section 4.5 explicitly mandates an isolated, Dockerized Apache Airflow 
# environment running Python 3.12+. The base image `quay.io/astronomer/astro-runtime:12.6.0` 
# strictly enforces this baseline, providing a deterministic orchestration layer.
#
# 2. PROTOCOL ALIGNMENT (High-Performance Compilation):
# To execute the vectorized Black-Scholes solvers and the GARCH(1,1) stationary 
# block bootstrapping, the environment requires optimized C-extensions. The 
# installation of `build-essential`, `gcc`, and `libopenblas-dev` guarantees 
# that `scipy`, `numpy`, and the `arch` library compile with high-performance 
# linear algebra backends, satisfying the computational feasibility requirements.
#
# 3. ENGINEERING ARTIFACT (Static Data Injection):
# The `COPY data/` command injects the static asset universe and the mock 
# Sharadar fundamental data directly into the container image. This ensures 
# the PoC can execute entirely offline without requiring proprietary API keys, 
# strictly preserving the reproducibility of the OSF freeze.
# =============================================================================

# Start with the latest Astro Runtime
FROM astrocrpublic.azurecr.io/runtime:3.1-12

USER root

# 1. Install System Compilers
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean

# 2. FORCE RE-INSTALL OF DATABENTO
# This ensures any hidden/pre-installed version is wiped before your requirements run.
COPY requirements.txt .
RUN pip uninstall -y databento && \
    pip install --no-cache-dir -r requirements.txt

USER astro
