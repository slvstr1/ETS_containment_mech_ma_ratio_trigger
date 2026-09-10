import datetime
import io
import os
import numpy as np
import pandas as pd
from arch import arch_model

# Import our configuration loader
from utils import load_config

# Load global configuration
CONFIG = load_config()


# ==============================================================================
# 1. DATA INGESTION & PARSING
# ==============================================================================
def parse_eua_csv(source) -> pd.DataFrame:
    """
    Reads a single EUA dataset file or string buffer, dynamically detects
    the 'Date' header row, and extracts standardized date & price columns.
    """
    if isinstance(source, str) and os.path.exists(source):
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            top_lines = [f.readline() for _ in range(20)]
        buffer = source
    elif isinstance(source, str):
        top_lines = source.strip().split("\n")[:20]
        buffer = io.StringIO(source.strip())
    else:
        top_lines = [source.readline() for _ in range(20)]
        if hasattr(source, "seek"):
            source.seek(0)
        buffer = source

    header_idx = 0
    for i, line in enumerate(top_lines):
        if "date" in line.lower():
            header_idx = i
            break

    df = pd.read_csv(buffer, sep=r"\t+|\s{2,}|,", engine="python", header=header_idx)
    df.columns = [str(c).strip().lower() for c in df.columns]

    date_candidates = [c for c in df.columns if "date" in c or "datum" in c]
    date_col = date_candidates[0] if date_candidates else df.columns[0]
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    for col_type in ["primary", "secondary"]:
        candidates = [c for c in df.columns if col_type in c]
        target_col = f"{col_type}_market"
        if candidates:
            df[target_col] = pd.to_numeric(
                df[candidates[0]].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
        else:
            df[target_col] = np.nan

    df = df.dropna(subset=["date"]).sort_values("date")
    return df[["date", "primary_market", "secondary_market"]]


def merge_eua_datasets(*dfs, start_year: int = None) -> pd.DataFrame:
    """
    Concatenates multiple EUA DataFrames, removes duplicates, filters by
    start_year (optional), and builds a unified daily price series.
    """
    df_merged = pd.concat(dfs, ignore_index=True)
    df_merged = df_merged.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    if start_year is not None:
        df_merged = df_merged[df_merged["date"].dt.year >= int(start_year)].reset_index(drop=True)

    # Coalesce primary and secondary markets into a single price column
    df_merged["price"] = df_merged["primary_market"].combine_first(df_merged["secondary_market"])

    # Forward-fill and backward-fill missing prices for continuous daily series
    df_merged["price"] = df_merged["price"].ffill().bfill()

    return df_merged


def export_dataset(df: pd.DataFrame, output_filepath: str):
    """Saves a DataFrame to disk, creating directories as needed."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"[I/O] Saved daily dataset CSV to: {output_filepath}")


# ==============================================================================
# 2. DAILY TIME SERIES PREPARATION (DAILY RETURNS)
# ==============================================================================
def prepare_daily_returns(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily log-returns y_t = ln(S_t / S_{t-1})."""
    df_work = df_daily.copy()
    df_work["daily_log_return"] = np.log(df_work["price"] / df_work["price"].shift(1))
    return df_work.dropna(subset=["daily_log_return"])


# ==============================================================================
# 3. DAILY MODELING & PARAMETER ESTIMATION (GARCH & LOGNORMAL)
# ==============================================================================
def estimate_daily_ar_garch_factors(daily_returns_series: pd.Series, days_per_month: int = 21) -> dict:
    """
    Fits AR(1)-GARCH(1,1) model to DAILY returns and extracts model criteria
    (LLF, AIC, BIC) and implied volatility metrics.
    """
    scaled_returns = daily_returns_series * 100.0

    model = arch_model(scaled_returns, mean="AR", lags=1, vol="Garch", p=1, q=1, dist="normal")
    res = model.fit(disp="off")

    llf = res.loglikelihood
    aic = res.aic
    bic = res.bic
    num_params = len(res.params)

    params_dict = {k.lower(): k for k in res.params.index}

    const_key = params_dict.get("const", params_dict.get("mu", res.params.index[0]))
    c_daily = res.params[const_key] / 100.0

    ar_keys = [k for k in res.params.index if "[t-1]" in k]
    phi_daily = res.params[ar_keys[0]] if ar_keys else 0.0

    omega_key = params_dict.get("omega", [k for k in res.params.index if "omega" in k.lower()][0])
    alpha_key = [k for k in res.params.index if "alpha" in k.lower()][0]
    beta_key = [k for k in res.params.index if "beta" in k.lower()][0]

    omega_daily = res.params[omega_key] / (100.0 ** 2)
    alpha = res.params[alpha_key]
    beta = res.params[beta_key]

    uncond_daily_var = (
        omega_daily / (1.0 - alpha - beta)
        if (alpha + beta < 1.0)
        else daily_returns_series.var()
    )
    daily_vol = np.sqrt(uncond_daily_var)

    days_per_year = days_per_month * 12
    implied_monthly_vol = daily_vol * np.sqrt(days_per_month) * 100.0
    implied_annual_vol = daily_vol * np.sqrt(days_per_year) * 100.0

    c_monthly = c_daily * float(days_per_month)
    target_mu_m = c_monthly / (1.0 - phi_daily) if abs(1.0 - phi_daily) > 1e-5 else c_monthly
    G_m = np.exp(target_mu_m)
    G_yearly = G_m ** 12

    return {
        "model_name": "AR(1)-GARCH(1,1)",
        "simulation_process": 0,
        "model_criteria": {
            "LLF (Log-Likelihood)": llf, "AIC": aic, "BIC": bic,
            "Num_Parameters": num_params, "Observations": len(daily_returns_series),
        },
        "daily_params": {
            "c_daily": c_daily, "phi_daily": phi_daily, "omega_daily": omega_daily,
            "alpha": alpha, "beta": beta, "daily_vol_%": daily_vol * 100.0,
        },
        "implied_benchmarks": {
            "c_monthly_equivalent": c_monthly, "G_m": G_m, "G_yearly": G_yearly,
            "implied_monthly_vol_%": implied_monthly_vol, "implied_annual_vol_%": implied_annual_vol,
        },
        "model_result": res,
    }


def estimate_daily_lognormal_factors(daily_returns_series: pd.Series, days_per_month: int = 21) -> dict:
    """
    Fits Constant Volatility Lognormal / Geometric Brownian Motion (GBM) model
    ln(P_t) - ln(P_{t-1}) = mu + sigma * eps_t to DAILY returns.
    """
    y = daily_returns_series.to_numpy()
    T = len(y)

    mu_daily = float(np.mean(y))
    sigma2_daily = float(np.var(y, ddof=0))  # MLE variance
    sigma_daily = float(np.sqrt(sigma2_daily))

    # Analytical Log-Likelihood for i.i.d. Normal:
    llf = -0.5 * T * np.log(2.0 * np.pi) - 0.5 * T * np.log(sigma2_daily) - 0.5 * T
    num_params = 2
    aic = 2.0 * num_params - 2.0 * llf
    bic = num_params * np.log(T) - 2.0 * llf

    days_per_year = days_per_month * 12
    implied_monthly_vol = sigma_daily * np.sqrt(days_per_month) * 100.0
    implied_annual_vol = sigma_daily * np.sqrt(days_per_year) * 100.0

    c_monthly = mu_daily * float(days_per_month)
    G_m = np.exp(c_monthly)
    G_yearly = G_m ** 12

    return {
        "model_name": "Lognormal Constant Volatility (GBM)",
        "simulation_process": 1,
        "model_criteria": {
            "LLF (Log-Likelihood)": llf, "AIC": aic, "BIC": bic,
            "Num_Parameters": num_params, "Observations": T,
        },
        "daily_params": {
            "mu_daily": mu_daily, "c_daily": mu_daily, "phi_daily": 0.0,
            "sigma_daily": sigma_daily, "omega_daily": sigma2_daily,
            "alpha": 0.0, "beta": 0.0, "daily_vol_%": sigma_daily * 100.0,
        },
        "implied_benchmarks": {
            "c_monthly_equivalent": c_monthly, "G_m": G_m, "G_yearly": G_yearly,
            "implied_monthly_vol_%": implied_monthly_vol, "implied_annual_vol_%": implied_annual_vol,
        },
        "model_result": None,
    }


# ==============================================================================
# 4. REPORTING & FILE LOGGING
# ==============================================================================
def format_estimation_report(df_daily_returns: pd.DataFrame, metrics: dict, start_year: int = None) -> str:
    """Formats estimation metrics and model fit criteria into a text report."""
    returns = df_daily_returns["daily_log_return"]
    criteria = metrics["model_criteria"]
    daily = metrics["daily_params"]
    bench = metrics["implied_benchmarks"]
    model_name = metrics["model_name"]
    is_garch = (metrics["simulation_process"] == 0)

    lines = [
        f"[DAILY DATASET SUMMARY - Model: {model_name}]",
        f"  - Start Year Filter   : {start_year if start_year else 'None (All Years)'}",
        f"  - Daily observations  : {len(returns)}",
        f"  - Date range          : {df_daily_returns['date'].min().strftime('%Y-%m-%d')} to {df_daily_returns['date'].max().strftime('%Y-%m-%d')}",
        f"  - Sample Daily Return : {returns.mean():.6f}",
        f"  - Sample Daily Vol    : {returns.std() * 100:.2f}%",
        "\n=========================================================",
        "  MODEL SELECTION CRITERIA (Benz & Trück 2009 / Burnham & Anderson)",
        "=========================================================",
        f"  Log-Likelihood (LLF)  = {criteria['LLF (Log-Likelihood)']:.2f}  (Higher is better)",
        f"  Akaike Info (AIC)     = {criteria['AIC']:.2f}  (Lower is better)",
        f"  Schwarz Bayesian (BIC)= {criteria['BIC']:.2f}  (Lower is better - Primary Metric)",
        f"  Estimated Parameters  = {criteria['Num_Parameters']}",
        "\n=========================================================",
        f"  ESTIMATED DAILY PARAMETERS ({model_name})",
        "=========================================================",
    ]

    if is_garch:
        lines.extend([
            f"  c (Daily Intercept)       = {daily['c_daily']:.6f}",
            f"  phi (Daily Autocorr)      = {daily['phi_daily']:.6f}",
            f"  omega / k (Base Variance) = {daily['omega_daily']:.10f}",
            f"  alpha (Shock Reaction)    = {daily['alpha']:.6f}",
            f"  beta (Var Persistence)    = {daily['beta']:.6f}",
            f"  alpha + beta (Persistence)= {(daily['alpha'] + daily['beta']):.6f}",
            f"  Unconditional Daily Vol   = {daily['daily_vol_%']:.4f}%",
        ])
    else:
        lines.extend([
            f"  mu (Daily Drift)          = {daily['mu_daily']:.6f}",
            f"  sigma (Daily Volatility)  = {daily['sigma_daily']:.6f} ({daily['daily_vol_%']:.4f}%)",
            f"  sigma^2 (Daily Variance)  = {daily['omega_daily']:.10f}",
        ])

    lines.extend([
        "\n=========================================================",
        "  IMPLIED MONTHLY & ANNUAL BENCHMARKS (For Simulation)",
        "=========================================================",
        f"  Implied Monthly Intercept (c) = {bench['c_monthly_equivalent']:.6f}",
        f"  Derived Monthly Growth (G_m)  = {bench['G_m']:.5f}",
        f"  Implied Yearly Growth Factor  = {bench['G_yearly']:.3f}x (+{(bench['G_yearly'] - 1) * 100:.1f}%/yr)",
        f"  Implied Monthly Volatility    = {bench['implied_monthly_vol_%']:.2f}%",
        f"  Implied Annual Volatility     = {bench['implied_annual_vol_%']:.2f}%",
        "========================================================="
    ])
    return "\n".join(lines)


def append_report_to_file(report_text: str, output_txt_filepath: str):
    """Appends formatted report text with a timestamp header to a log file."""
    os.makedirs(os.path.dirname(output_txt_filepath), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_txt_filepath, "a", encoding="utf-8") as f:
        f.write(f"\n{'#' * 65}\n")
        f.write(f"# DAILY ESTIMATION RUN TIMESTAMP: {timestamp}\n")
        f.write(f"{'#' * 65}\n")
        f.write(report_text)
        f.write("\n\n")

    print(f"[I/O] Appended results to: {output_txt_filepath}")


# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    data_cfg = CONFIG["data"]
    paths_cfg = CONFIG["paths"]
    dpm = CONFIG["stochastic"].get("days_per_month", 21)

    # Detect simulation process: 0 = AR(1)-GARCH(1,1), 1 = Constant Volatility Lognormal
    sim_proc = CONFIG.get("simulation", {}).get("simulation_process", 0)

    file_2010_2018 = data_cfg["raw_2010_2018"]
    file_2019_2026 = data_cfg["raw_2019_2026"]
    start_year = data_cfg["start_year"]

    out_dir = paths_cfg["output_data_dir"]
    output_csv = os.path.join(out_dir, f"merged_eua_{start_year}_2026.csv")
    output_txt = os.path.join(out_dir, f"estimation_results_{start_year}_2026.txt")

    if not os.path.exists(file_2010_2018) or not os.path.exists(file_2019_2026):
        raise FileNotFoundError(f"Missing one of the raw data files in {data_cfg}")

    print(f"Parsing input files starting from year {start_year}...")
    df1 = parse_eua_csv(file_2010_2018)
    df2 = parse_eua_csv(file_2019_2026)

    # 1. Merge datasets
    df_daily = merge_eua_datasets(df1, df2, start_year=start_year)
    export_dataset(df_daily, output_csv)

    # 2. Prepare returns & estimate parameters
    df_daily_returns = prepare_daily_returns(df_daily)

    if sim_proc == 1:
        print("Estimating Constant Volatility Lognormal (GBM) parameters (simulation_process = 1)...")
        metrics = estimate_daily_lognormal_factors(df_daily_returns["daily_log_return"], days_per_month=dpm)
    else:
        print("Estimating AR(1)-GARCH(1,1) parameters (simulation_process = 0)...")
        metrics = estimate_daily_ar_garch_factors(df_daily_returns["daily_log_return"], days_per_month=dpm)

    # 3. Output report
    report_text = format_estimation_report(df_daily_returns, metrics, start_year=start_year)
    print("\n" + report_text + "\n")
    append_report_to_file(report_text, output_txt)

    # 4. Convenient TOML summary for config.toml
    p = metrics["daily_params"]
    b = metrics["implied_benchmarks"]
    print("---------------------------------------------------------")
    print("To use these estimated factors in config.toml, verify:")
    print(f"  simulation_process = {sim_proc}")
    if sim_proc == 1:
        print(f"  sigma_daily = {p['sigma_daily']:.6f}")
    else:
        print(f"  alpha = {p['alpha']:.6f}")
        print(f"  beta  = {p['beta']:.6f}")
        print(f"  phi_daily = {p['phi_daily']:.6f}")
    print(f"  empirical_monthly_{start_year}_x = {b['G_m']:.5f}")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    main()