# EU ETS Price Containment Mechanism: Moving Average (MA) Ratio Trigger Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Managed with uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical and stochastic simulation framework designed to evaluate the statistical performance, trigger probabilities, and error rates of **price containment and market stability mechanisms** in the **European Union Emissions Trading System (EU ETS)**.

---

## 📌 Background & Regulatory Context

Under EU ETS legislation, excessive price spikes trigger market intervention (such as releasing additional European Union Allowances (EUAs) from the Market Stability Reserve):

* **EU ETS 1 (Directive 2003/87/EC, Article 29a):**  
  Measures whether the average price of the preceding **$p_2 = 6$ consecutive months** is more than **$m = 2.4$ times** the average price of the preceding **$p_1 = 24$ months** (two years).
* **EU ETS 2 (Directive 2003/87/EC, Article 30h):**  
  Measures whether the average price of the preceding **$p_2 = 3$ consecutive months** exceeds **$m = 1.5$ times** (or $m = 2.0$ in subsequent phases) the average price of the preceding **$p_1 = 6$ months**.
* **Historical ETS 1 (Pre-reform Article 29a):**  
  Prior regime with window parameters $p_1 = 6$, $p_2 = 3$, and threshold multiplier $m = 3.0$.

### Moving Average Ratio Formula
At any month $t$, the non-overlapping moving average ratio $R_t$ is defined as:

$$R_t = \frac{\mathrm{MA}_{p_2}(t)}{\mathrm{MA}_{p_1}(t - p_2)} = \frac{\frac{1}{p_2} \sum_{i=0}^{p_2-1} P_{t-i}}{\frac{1}{p_1} \sum_{j=0}^{p_1-1} P_{t-p_2-j}}$$

An intervention is triggered if $R_t \ge m$ (price surge) or, symmetrically, if $R_t \le 1/m$ (price collapse).

This repository evaluates whether these statutory filters successfully detect true structural drifts ($\text{True Positives}$) versus market noise ($\text{False Triggers}$ or $\text{Trigger Misses}$).

---

## 🚀 Key Features

* **Empirical Parameter Estimation (`01_estimate_eua_factors.py`):**
  * Ingests primary auction and secondary market ICAP historical EUA price series (2010–2026).
  * Estimates both **AR(1)-GARCH(1,1)** (conditional heteroskedasticity) and **Geometric Brownian Motion (GBM)** models via Maximum Likelihood Estimation (MLE).
  * Computes log-likelihood (LLF), Akaike Information Criterion (AIC), and Schwarz Bayesian Information Criterion (BIC).
* **High-Performance Monte Carlo Engine (`02__*.py`):**
  * Multiprocessed simulation across a continuous grid of target annual price drifts $X$ and monthly volatilities $\sigma_m$.
  * Binned caching system for rapid re-plotting without re-simulating millions of sample paths.
  * Comprehensive diagnostic metric suite: Trigger Probability, Trigger Miss / False Trigger rates ($\pi, \rho$), Miss Ratio ($\eta$), False Trigger Ratio ($\kappa$), and Parity deviations.
* **Historical Diagnostics (`03_Historical Plot...py`):**
  * Reconstructs historical $R_t$ trajectory against statutory boundaries ($m \in \{1.5, 2.0, 2.4, 3.0\}$).
  * Multi-panel visualization supporting daily/monthly price series, $\ln(\text{Price})$, and regime change dates.
* **Trajectory Inspection (`04_plot_sample_paths.py`):**
  * Plots synthetic asset price trajectories alongside corresponding rolling ratios and trigger points.

---

## 📂 Project Architecture

```text
├── config.toml                     # Master configuration (paths, regimes, toggles, styles)
├── data_input/                     # Raw historical EUA datasets
│   ├── icap-uea-price-daily-2010-2018.csv
│   └── icap-uea-price-daily-2019-2026.csv
├── data_output/                    # Processed CSVs, estimation logs, and SVG figures
├── utils.py                        # Safe configuration loader & path resolver
├── 01_estimate_eua_factors.py      # Historical econometric calibration
├── 02__generate_simulations.py     # Monte Carlo batch simulator
├── 02__draw_simulations.py         # Monte Carlo results visualizer
├── 03_Historical Plot of Prices and MA Ratios.py  # Historical empirical ratio plots
├── 04_plot_sample_paths.py         # Synthetic sample path generator
├── simulation_engine.py            # Simulation core (vectorized paths & triggers)
├── plot_engine.py                  # Matplotlib rendering engine
├── pyproject.toml                  # uv / PEP 621 dependency specifications
└── uv.lock                         # Locked dependency graph
