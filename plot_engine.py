import os\


# Set single-thread environment variables before importing numpy to prevent CPU over-subscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import datetime
import hashlib
import json
# import multiprocessing
import glob
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

from utils import load_config
from simulation_engine import (
    simulate_ar_garch_daily_batch,
    simulate_lognormal_daily_batch,
    get_daily_c_for_target_X,
    evaluate_ma_violations,
    evaluate_ma_violations_daily,
    solve_max_growth_r,
    log_warmup_diagnostics,
)

CONFIG = load_config()


# ==============================================================================
# CONFIGURATION DIAGNOSTICS PRINT
# ==============================================================================
def run_config_diagnostics() -> None:
    """Prints retrieved TOML configuration variables to verify paths are correct."""
    print("=" * 70)
    print(" CONFIGURATION DIAGNOSTICS")
    print("=" * 70)
    try:
        print("STOCHASTIC:")
        print(f"  alpha:          {CONFIG['stochastic'].get('alpha')}")
        print(f"  beta:           {CONFIG['stochastic'].get('beta')}")
        print(f"  days_per_month: {CONFIG['stochastic'].get('days_per_month')}")

        print("\nSIMULATION:")
        sim = CONFIG.get("simulation", {})
        print(f"  focus_metric:        {sim.get('focus_metric')}")
        print(f"  show_drift_parity:   {sim.get('show_drift_parity')}")
        print(f"  tail_direction:      {sim.get('tail_direction')}")
        print(f"  show_realized_tail:  {sim.get('show_realized_tail')}")
        print(f"  default_x_lower_min: {sim.get('default_x_lower_min')}")
        print(f"  default_x_lower_max: {sim.get('default_x_lower_max')}")
        print(f"  default_x_upper_min: {sim.get('default_x_upper_min')}")
        print(f"  default_x_upper_max: {sim.get('default_x_upper_max')}")

        print("\nPLOTTING:")
        plotting = CONFIG.get("plotting", {})
        print(f"  markersize (global): {plotting.get('markersize')}")
        print(f"  PLOT_COLORS:         {plotting.get('colors', {}).get('plot_colors')}")
        print(f"  PLOT_MARKERS:        {plotting.get('colors', {}).get('plot_markers')}")
        print(f"  GRID_CFG:            {plotting.get('grid')}")
        print(f"  FONTS_CFG:           {plotting.get('fonts')}")
        print(f"  LABELS_CFG:          {plotting.get('labels')}")

        print("\nPATHS:")
        print(f"  output_data_dir:     {CONFIG['paths'].get('output_data_dir')}")
        print(f"  output_figures:      {CONFIG['paths'].get('output_figures')}")
        print(f"  log_dir:             {CONFIG['paths'].get('log_dir')}")
    except Exception as e:
        print(f"  [ERROR] Failed to run configuration diagnostic print: {e}")
    print("=" * 70)


# Run config diagnostics on startup
run_config_diagnostics()

PLOT_COLORS = CONFIG["plotting"]["colors"]["plot_colors"]
PLOT_MARKERS = CONFIG["plotting"]["colors"]["plot_markers"]

FONTS_CFG = CONFIG["plotting"]["fonts"]
LABELS_CFG = CONFIG["plotting"]["labels"]
GRID_CFG = CONFIG["plotting"]["grid"]

METRIC_CONFIGS = {
    "conditional_count": {
        "col": "Avg_Violations_Conditional",
        "label": LABELS_CFG["y_axis_conditional_count"],
        "title": LABELS_CFG["title_count"],
        "tag": "_conditional_avg_count",
    },
    "unconditional_count": {
        "col": "Avg_Violations_Per_Series",
        "label": LABELS_CFG["y_axis_unconditional_count"],
        "title": LABELS_CFG["title_count"],
        "tag": "_unconditional_avg_count",
    },
    "probability": {
        "col": "Violation_Prob_%",
        "label": LABELS_CFG["y_axis_probability"],
        "title": LABELS_CFG["title_probability"],
        "tag": "",
    },
}


# ==============================================================================
# LOGGING & STYLE UTILITIES
# ==============================================================================
def log_message(message: str, filename: str = "log_theory_drift.txt") -> None:
    """Appends a timestamped log line to file and flushes to console."""
    log_dir = CONFIG["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_line = f"[{timestamp}] {message}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(full_line)
    print(full_line, end="", flush=True)


def get_volatility_style(vol: float, master_positive_vols: Optional[List[float]] = None) -> Tuple[str, str]:
    """
    Returns (color, marker) ensuring strict indexing consistency:
      - 0.0% volatility is ALWAYS index 0.
      - Positive volatilities are ALWAYS indices 1, 2, 3, ... (regardless of whether 0% is present).
    """
    if np.isclose(vol, 0.0):
        style_idx = 0
    else:
        if master_positive_vols and vol in master_positive_vols:
            pos_rank = master_positive_vols.index(vol)
        else:
            pos_rank = 0
        style_idx = 1 + pos_rank

    c = PLOT_COLORS[style_idx % len(PLOT_COLORS)]
    m = PLOT_MARKERS[style_idx % len(PLOT_MARKERS)]
    return c, m


# ==============================================================================
# ANALYTICAL SWITCH POINT SOLVER
# ==============================================================================
def get_critical_X(p1: int, p2: int, m: float, mode: str, eval_frequency: str, days_per_month: int) -> float:
    """Calculates the analytical threshold annualized growth factor X_crit where the transition occurs."""
    detect_reduction = (mode == "reduction")
    m_target = (1.0 / m) if detect_reduction else m

    if eval_frequency == "daily":
        D1 = p1 * days_per_month
        D2 = p2 * days_per_month
        r_daily = solve_max_growth_r(D1, D2, m_target)
        return float(r_daily ** (12 * days_per_month))
    else:
        r_monthly = solve_max_growth_r(p1, p2, m_target)
        return float(r_monthly ** 12.0)


# ==============================================================================
# PRICE SERIES CACHING & RETRIEVAL ENGINE
# ==============================================================================
def get_price_series_cached(
        p1: int, p2: int, vol_pct: float, n_eval_months: int, S0: float,
        c_daily: float, sigma_daily: float, eff_phi: float,
        omega_daily: float, eff_alpha: float, eff_beta: float,
        days_per_month: int, is_daily_eval: bool,
        warm_up_max: bool, warm_up_r: float, n_warm_up_months: int,
        sim_proc: int, n_sims: int, cache_dir: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Retrieves generated price paths from cache, or simulates remaining/all paths,
    supporting differential simulation and chunked recombination logic.
    """
    price_cache_dir = os.path.join(cache_dir, "price_series")
    os.makedirs(price_cache_dir, exist_ok=True)
    N_total_months = p1 + p2 + n_eval_months

    base_params = {
        "p1": p1,
        "p2": p2,
        "vol_pct": vol_pct,
        "n_eval_months": n_eval_months,
        "S0": S0,
        "c_daily": round(c_daily, 12),
        "sigma_daily": round(sigma_daily, 12),
        "eff_phi": round(eff_phi, 12),
        "omega_daily": round(omega_daily, 12),
        "eff_alpha": round(eff_alpha, 12),
        "eff_beta": round(eff_beta, 12),
        "days_per_month": days_per_month,
        "is_daily_eval": is_daily_eval,
        "warm_up_max": warm_up_max,
        "warm_up_r": round(warm_up_r, 12),
        "n_warm_up_months": n_warm_up_months,
        "sim_proc": sim_proc
    }
    base_key = hashlib.md5(json.dumps(base_params, sort_keys=True).encode("utf-8")).hexdigest()
    vol_str = f"{vol_pct:.4f}"

    pattern = os.path.join(price_cache_dir, f"prices_p1_{p1}_p2_{p2}_vol_{vol_str}_nsims_*_{base_key}.npz")
    matching_files = glob.glob(pattern)

    file_infos = []
    regex = re.compile(rf"prices_p1_{p1}_p2_{p2}_vol_{re.escape(vol_str)}_nsims_(\d+)_{base_key}\.npz$")
    for fpath in matching_files:
        fname = os.path.basename(fpath)
        match = regex.match(fname)
        if match:
            nsims_val = int(match.group(1))
            file_infos.append((fpath, nsims_val))

    file_infos.sort(key=lambda x: x[1], reverse=True)

    selected_files = []
    accumulated_count = 0
    for fpath, nsims_val in file_infos:
        if accumulated_count < n_sims:
            selected_files.append((fpath, nsims_val))
            accumulated_count += nsims_val
            if accumulated_count >= n_sims:
                break

    needed_monthly = None
    needed_daily = None

    if accumulated_count < n_sims:
        n_needed = n_sims - accumulated_count
        if sim_proc == 1:
            sim_res = simulate_lognormal_daily_batch(
                N_months=N_total_months, batch_size=n_needed, S0=S0,
                mu_daily=c_daily, sigma_daily=sigma_daily, days_per_month=days_per_month,
                return_daily=is_daily_eval, warm_up_max=warm_up_max,
                warm_up_r=warm_up_r, n_warm_up_months=n_warm_up_months
            )
        else:
            sim_res = simulate_ar_garch_daily_batch(
                N_months=N_total_months, batch_size=n_needed, S0=S0,
                c_daily=c_daily, phi_daily=eff_phi, omega_daily=omega_daily,
                alpha=eff_alpha, beta=eff_beta, days_per_month=days_per_month,
                return_daily=is_daily_eval, warm_up_max=warm_up_max,
                warm_up_r=warm_up_r, n_warm_up_months=n_warm_up_months
            )

        if is_daily_eval:
            needed_monthly, needed_daily = sim_res
        else:
            needed_monthly = sim_res
            needed_daily = None

        new_fpath = os.path.join(price_cache_dir,
                                 f"prices_p1_{p1}_p2_{p2}_vol_{vol_str}_nsims_{n_needed}_{base_key}.npz")
        if is_daily_eval:
            np.savez_compressed(new_fpath, S_monthly_avg=needed_monthly, S_daily=needed_daily)
        else:
            np.savez_compressed(new_fpath, S_monthly_avg=needed_monthly)

    loaded_monthly = []
    loaded_daily = []
    for fpath, _ in selected_files:
        with np.load(fpath) as data:
            loaded_monthly.append(np.array(data["S_monthly_avg"]))
            if is_daily_eval:
                loaded_daily.append(np.array(data["S_daily"]))

    if needed_monthly is not None:
        loaded_monthly.append(needed_monthly)
    if is_daily_eval and needed_daily is not None:
        loaded_daily.append(needed_daily)

    if len(loaded_monthly) > 0:
        S_monthly_avg = np.concatenate(loaded_monthly, axis=1)[:, :n_sims]
        if is_daily_eval:
            S_daily = np.concatenate(loaded_daily, axis=1)[:, :n_sims]
        else:
            S_daily = None
    else:
        raise ValueError("No price series files could be constructed, loaded, or simulated.")

    return S_monthly_avg, S_daily


# ==============================================================================
# STREAMING SIMULATION TASK ENGINE
# ==============================================================================
# ==============================================================================
# TRUE STREAMING SIMULATION TASK ENGINE (LOW MEMORY FOOTPRINT)
# ==============================================================================
def eval_theory_grid_point_streaming(task_tuple: tuple) -> Dict[str, Any]:
    """Generates price paths strictly in small batches, accumulating statistics on the fly."""
    (
        x_target_val, x_cum, v, p1, p2, m, n_sims, batch_size,
        n_eval_months, phi_daily, eff_alpha, eff_beta, days_per_month,
        eval_frequency, warm_up_max, sim_proc, tail_direction, mode
    ) = task_tuple

    delta_M = n_eval_months
    N_total_months = p1 + p2 + delta_M
    years = delta_M / 12.0
    detect_reduction = (mode == "reduction")

    eff_phi = 0.0 if sim_proc == 1 else phi_daily
    c_daily, sigma_daily = get_daily_c_for_target_X(x_cum, delta_M, v, eff_phi, days_per_month)
    omega_daily = (sigma_daily ** 2) * (1.0 - eff_alpha - eff_beta)

    warm_up_r = solve_max_growth_r(p1, p2, (1.0 / m) if detect_reduction else m) if warm_up_max else 1.0
    n_warm_up_months = (p1 + p2 - 1) if warm_up_max else 0

    num_batches = int(np.ceil(n_sims / batch_size))
    is_daily_eval = (eval_frequency == "daily")

    # Streaming Accumulators (Only scalars kept in memory)
    total_paths = total_triggered = total_upper = total_lower = 0
    total_tp = total_fp = total_fn = total_tn = total_violations_sum = 0

    for b in range(num_batches):
        cur_batch = min(batch_size, n_sims - (b * batch_size))
        if cur_batch <= 0:
            break

        # 1. Simulate ONLY cur_batch (e.g. 50,000) paths
        if sim_proc == 1:
            sim_res = simulate_lognormal_daily_batch(
                N_months=N_total_months, batch_size=cur_batch, S0=100.0,
                mu_daily=c_daily, sigma_daily=sigma_daily, days_per_month=days_per_month,
                return_daily=is_daily_eval, warm_up_max=warm_up_max,
                warm_up_r=warm_up_r, n_warm_up_months=n_warm_up_months
            )
        else:
            sim_res = simulate_ar_garch_daily_batch(
                N_months=N_total_months, batch_size=cur_batch, S0=100.0,
                c_daily=c_daily, phi_daily=eff_phi, omega_daily=omega_daily,
                alpha=eff_alpha, beta=eff_beta, days_per_month=days_per_month,
                return_daily=is_daily_eval, warm_up_max=warm_up_max,
                warm_up_r=warm_up_r, n_warm_up_months=n_warm_up_months
            )

        if is_daily_eval:
            batch_S_monthly_avg, batch_S_daily = sim_res
            violation_counts = evaluate_ma_violations_daily(
                S_daily=batch_S_daily, p1=p1, p2=p2, m=m,
                days_per_month=days_per_month, detect_reduction=detect_reduction
            )
            del batch_S_daily  # Free heavy daily array immediately
        else:
            batch_S_monthly_avg = sim_res
            violation_counts = evaluate_ma_violations(
                S=batch_S_monthly_avg, p1=p1, p2=p2, m=m, detect_reduction=detect_reduction
            )

        # 2. Evaluate growth & violations for this batch
        ratios = batch_S_monthly_avg[-1, :] / batch_S_monthly_avg[p1 + p2 - 1, :]
        yearly_growths = ratios ** (1.0 / years)

        triggered_mask = (violation_counts > 0)
        upper_mask = (yearly_growths >= (x_target_val - 1e-7))
        lower_mask = (yearly_growths < (x_target_val - 1e-7))

        # 3. Accumulate scalar stats
        total_paths += cur_batch
        total_triggered += int(np.sum(triggered_mask))
        total_upper += int(np.sum(upper_mask))
        total_lower += int(np.sum(lower_mask))
        total_tp += int(np.sum(triggered_mask & upper_mask))
        total_fp += int(np.sum(triggered_mask & lower_mask))
        total_fn += int(np.sum((~triggered_mask) & upper_mask))
        total_tn += int(np.sum((~triggered_mask) & lower_mask))
        total_violations_sum += int(np.sum(violation_counts))

        # 4. Clean up batch memory
        del batch_S_monthly_avg, violation_counts

    # Evaluate metric ratios cleanly
    prob_viol = float((total_triggered / total_paths) * 100.0)
    prob_upper = float((total_upper / total_paths) * 100.0)
    prob_lower = float((total_lower / total_paths) * 100.0)

    return {
        "Monthly_Vol_%": v,
        "Base_Variance_omega_daily": round(omega_daily, 10),
        "Target_Yearly_X": round(float(x_target_val), 4),
        "Path_Count": total_paths,
        "Violation_Prob_%": round(prob_viol, 3),
        "Realized_Upper_Prob_%": round(prob_upper, 3),
        "Realized_Lower_Prob_%": round(prob_lower, 3),
        "Trigger_Miss_%": round(float((total_fn / total_paths) * 100.0), 3),
        "False_Trigger_%": round(float((total_fp / total_paths) * 100.0), 3),
        "Relative_Trigger_Ratio": round(float(prob_viol / prob_upper) if prob_upper > 0.0 else np.nan, 4),
        "Miss_Ratio": round(float(total_fn / total_upper) if total_upper > 0 else 0.0, 4),
        "False_Trigger_Ratio": round(float(total_fp / total_lower) if total_lower > 0 else 0.0, 4),
        "True_Positive_%": round(float((total_tp / total_paths) * 100.0), 3),
        "True_Negative_%": round(float((total_tn / total_paths) * 100.0), 3),
        "Avg_Violations_Per_Series": round(float(total_violations_sum / total_paths), 4),
        "Avg_Violations_Conditional": round(float(total_violations_sum / total_triggered), 4) if total_triggered > 0 else np.nan,
    }


# ==============================================================================
# DATA GENERATION & CACHING
# ==============================================================================
def generate_theoretical_drift_data_cached(
        cfg: dict, vol_list: List[float], phi_daily: float, n_sims: int, batch_size: int,
        n_eval_months: int, n_jobs: int, use_cache: bool, cache_dir: str,
        eval_frequency: str = "monthly", sim_proc: int = 0, tail_direction: str = "upper"
) -> pd.DataFrame:
    """Runs parallel simulations across (volatility, target) grid or loads cached parquet."""
    os.makedirs(cache_dir, exist_ok=True)
    mode, m, p1, p2 = cfg["mode"], cfg["m"], cfg["p1"], cfg["p2"]
    x_min, x_max, num_x_points = cfg["x_min"], cfg["x_max"], cfg.get("num_x_points", 200)
    warm_up_max = cfg.get("warm_up_max", False)

    eff_phi = 0.0 if sim_proc == 1 else phi_daily
    eff_alpha = 0.0 if sim_proc == 1 else cfg["alpha"]
    eff_beta = 0.0 if sim_proc == 1 else cfg["beta"]
    dpm = cfg["days_per_month"]

    cache_kwargs = {
        "sim_proc": sim_proc, "tail_direction": tail_direction, "mode": mode,
        "vol_list": vol_list, "phi": eff_phi, "alpha": eff_alpha, "beta": eff_beta,
        "x_min": x_min, "x_max": x_max, "num_x_points": num_x_points,
        "p1": p1, "p2": p2, "m": m, "n_sims": n_sims, "n_eval_months": n_eval_months,
        "eval_freq": eval_frequency, "warm_up_max": warm_up_max
    }
    cache_key = hashlib.md5(json.dumps(cache_kwargs, sort_keys=True).encode("utf-8")).hexdigest()
    proc_tag = "lognorm" if sim_proc == 1 else "garch"
    cache_file = os.path.join(cache_dir, f"vol_{proc_tag}_theory_{tail_direction}_{mode}_{cache_key}.parquet")

    proc_label = "LOGNORMAL CONST-VOL" if sim_proc == 1 else "AR(1)-GARCH(1,1)"
    mode_label = f"THEORY-BASED ({tail_direction.upper()} TAIL)"

    if use_cache and os.path.exists(cache_file):
        df_cached = pd.read_parquet(cache_file)
        required_schema = {"Violation_Prob_%", "Trigger_Miss_%", "False_Trigger_%", "Miss_Ratio", "False_Trigger_Ratio"}
        if required_schema.issubset(df_cached.columns):
            log_message(
                f"[CACHE HIT] Loading ({proc_label} | {mode_label} | {mode.upper()}, m={m}, p1={p1}, p2={p2}): {cache_file}")
            return df_cached
        log_message(
            f"[CACHE OUTDATED] Re-simulating to populate Trigger Miss & False Trigger metrics for {mode.upper()}...")

    years = n_eval_months / 12.0
    x_yearly_targets = np.linspace(x_min, x_max, num_x_points)

    x_crit = get_critical_X(p1, p2, m, mode, eval_frequency, dpm)
    if x_min <= x_crit <= x_max:
        micro_offsets = np.array([x_crit - 1e-5, x_crit, x_crit + 1e-5])
        x_yearly_targets = np.unique(np.sort(np.concatenate([x_yearly_targets, micro_offsets])))

    x_cumulative_targets = x_yearly_targets ** years

    tasks = [
        (
            x_val, x_cum, v, p1, p2, m, n_sims, batch_size,
            n_eval_months, eff_phi, eff_alpha, eff_beta, dpm,
            eval_frequency, warm_up_max, sim_proc, tail_direction, mode
        )
        for v in vol_list for x_val, x_cum in zip(x_yearly_targets, x_cumulative_targets)
    ]

    log_message(
        f"[RUN START] Model={proc_label} | Grid={mode_label} | Mode={mode.upper()} | Freq={eval_frequency.upper()} | WarmUp={'MAX' if warm_up_max else 'NONE'} | Window=[{x_min}, {x_max}] | m={m}")

    results = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(eval_theory_grid_point_streaming)(t) for t in tasks
    )

    df_out = pd.DataFrame(results)
    df_out["Model"] = "Lognormal" if sim_proc == 1 else "AR-GARCH"
    df_out["Mode"] = mode
    df_out["Tail_Direction"] = tail_direction
    df_out["m"] = m
    df_out["p1"] = p1
    df_out["p2"] = p2
    df_out["Warm_Up_Max"] = warm_up_max
    df_out["Eval_Frequency"] = eval_frequency
    df_out["N_Eval_Months"] = n_eval_months

    df_out.to_parquet(cache_file, index=False)
    log_message(f"[RUN END] Model={proc_label} | Grid={mode_label} | Mode={mode.upper()} -> Saved to {cache_file}")
    return df_out


# ==============================================================================
# DATABASE PERSISTENCE
# ==============================================================================
def _update_master_database(df_graph: pd.DataFrame, cfg: dict, sim_proc: int, tail_direction: str) -> None:
    """Updates master CSV database with simulation results."""
    master_csv = os.path.join(CONFIG["paths"]["output_data_dir"], "master_relative_trigger_metrics.csv")
    required_cols = {"m", "p1", "p2", "Mode", "Tail_Direction", "Warm_Up_Max", "Model"}
    model_str = "Lognormal" if sim_proc == 1 else "AR-GARCH"

    if os.path.exists(master_csv):
        try:
            df_existing = pd.read_csv(master_csv)
            if required_cols.issubset(df_existing.columns):
                mask_dup = (
                        (df_existing["m"] == cfg["m"]) &
                        (df_existing["p1"] == cfg["p1"]) &
                        (df_existing["p2"] == cfg["p2"]) &
                        (df_existing["Mode"] == cfg["mode"]) &
                        (df_existing["Tail_Direction"] == tail_direction) &
                        (df_existing["Warm_Up_Max"] == cfg.get("warm_up_max", False)) &
                        (df_existing["Model"] == model_str)
                )
                df_combined = pd.concat([df_existing[~mask_dup], df_graph], ignore_index=True)
                df_combined.to_csv(master_csv, index=False)
            else:
                df_graph.to_csv(master_csv, index=False)
        except Exception:
            df_graph.to_csv(master_csv, index=False)
    else:
        df_graph.to_csv(master_csv, index=False)

    log_message(f"[DATABASE] Updated master table -> {master_csv}")


# ==============================================================================
# MODULAR PLOTTING HELPERS
# ==============================================================================
def _apply_grid_and_ticks(ax: plt.Axes, cfg: dict) -> None:
    """Applies major and minor grid styling and locators from config."""
    ax.locator_params(axis="x", nbins=cfg.get("x_nbins", 10))

    if GRID_CFG["show_major_grid"]:
        ax.grid(
            True, which="major",
            linestyle=GRID_CFG["major_grid_linestyle"],
            alpha=float(GRID_CFG["major_grid_alpha"]),
            linewidth=float(GRID_CFG["major_grid_width"]),
            color=GRID_CFG["major_grid_color"]
        )

    if GRID_CFG["show_minor_grid"]:
        subdivisions = GRID_CFG["minor_subdivisions"]
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(subdivisions))
        ax.grid(
            True, which="minor",
            linestyle=GRID_CFG["minor_grid_linestyle"],
            alpha=float(GRID_CFG["minor_grid_alpha"]),
            linewidth=float(GRID_CFG["minor_grid_width"]),
            color=GRID_CFG["minor_grid_color"]
        )
    else:
        ax.minorticks_off()


def _build_legend_artists(
        vols: List[float], tail_direction: str = "upper", distinguish_tp_fp: bool = False,
        focus_metric: str = "trigger", tail_only: bool = False,
        master_pos_vols: Optional[List[float]] = None
) -> Tuple[List, List, List, List, str, List, List, str]:
    """Constructs separated handles and labels for Volatility (left) and Metrics/Signals (right)."""
    vol_handles, vol_labels = [], []
    vol_template = LABELS_CFG["label_vol_entry"]
    vol_title = LABELS_CFG["legend_vol_title"]

    tail_line_cfg = CONFIG["plotting"]["tail_reference_line"]
    tail_linestyle = tail_line_cfg["linestyle"]
    tail_linewidth = float(tail_line_cfg["linewidth"])

    # 1. Build Volatility Handles (Left Box)
    for v in vols:
        c, m = get_volatility_style(v, master_pos_vols)
        proxy = Line2D(
            [0], [0], color=c,
            marker=m, markevery=1, linewidth=2.5, markersize=8
        )
        vol_handles.append(proxy)
        if "{v" in vol_template:
            formatted_v = vol_template.replace("{v:.1f}", f"{v:.1f}").replace("{v}", f"{v:.1f}").replace("%", "").strip()
            vol_labels.append(fr"${formatted_v}$" + "%")
        else:
            vol_labels.append(fr"$\sigma_m$ = {v:.1f}%")

    vol_handles = vol_handles[::-1]
    vol_labels = vol_labels[::-1]

    if tail_only:
        return vol_handles, vol_labels, vol_handles, vol_labels, vol_title, [], [], ""

    # 2. Build Metric / Signal Style Handles (Right Box)
    style_handles, style_labels = [], []
    style_title = ""

    if distinguish_tp_fp:
        style_title = LABELS_CFG["legend_header_signals"]
        lbl_tp = LABELS_CFG["label_tp"]
        lbl_fp = LABELS_CFG["label_fp"]
        lbl_fn = LABELS_CFG["label_fn"]

        style_handles = [
            Line2D([0], [0], color="#222222", linestyle="-", linewidth=2.5, marker="o", markersize=6),
            Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.0),
            Line2D([0], [0], color="#222222", linestyle=":", linewidth=2.0)
        ]
        style_labels = [lbl_tp, lbl_fp, lbl_fn]
    elif focus_metric == "trigger_failure":
        style_title = LABELS_CFG["legend_header_metrics"]
        lbl_main = LABELS_CFG["label_false_trigger"] if tail_direction == "lower" else LABELS_CFG["label_trigger_miss"]
        lbl_tail = LABELS_CFG["label_tail_prob_lower"] if tail_direction == "lower" else LABELS_CFG["label_tail_prob_upper"]

        style_handles = [
            Line2D([0], [0], color="#222222", linestyle="-", linewidth=2.5, marker="o", markersize=6),
            Line2D([0], [0], color="#222222", linestyle=tail_linestyle, linewidth=tail_linewidth)
        ]
        style_labels = [lbl_main, lbl_tail]
    else:
        style_title = LABELS_CFG["legend_header_metrics"]
        lbl_triggered = LABELS_CFG["label_triggered"]
        lbl_tail = LABELS_CFG["label_tail_prob_upper"]

        style_handles = [
            Line2D([0], [0], color="#222222", linestyle="-", linewidth=2.5, marker="o", markersize=6),
            Line2D([0], [0], color="#222222", linestyle=tail_linestyle, linewidth=tail_linewidth)
        ]
        style_labels = [lbl_triggered, lbl_tail]

    # Combined lists for normal inside-plot legends
    combined_handles = vol_handles + [Line2D([], [], color="none")] + style_handles
    combined_labels = vol_labels + [f"\n{style_title}"] + style_labels

    return (
        combined_handles, combined_labels,
        vol_handles, vol_labels, vol_title,
        style_handles, style_labels, style_title
    )


def _save_standalone_legend(
        fig_main: plt.Figure,
        vol_handles: List, vol_labels: List, vol_title: str,
        save_prefix: str, sample_tag: str, output_folder: str,
        save_fig: bool
) -> None:
    """Exports a clean standalone legend containing ONLY Monthly Volatility."""
    plt.close(fig_main)
    fig_leg, ax_leg = plt.subplots(figsize=(4.8, 3.6), dpi=150)
    ax_leg.axis("off")

    leg_title_size = FONTS_CFG["legend_title_size"]
    leg_font_size = FONTS_CFG["legend_font_size"]

    leg = ax_leg.legend(
        vol_handles, vol_labels, title=vol_title,
        title_fontproperties={"weight": "bold", "size": leg_title_size},
        fontsize=leg_font_size, loc="center",
        markerscale=0.9, frameon=True, borderpad=0.4, labelspacing=0.35, handletextpad=0.5
    )

    if save_fig:
        for ext in ["svg", "png"]:
            path = f"{output_folder}{save_prefix}_legend_only_{sample_tag}.{ext}"
            fig_leg.savefig(
                path,
                format=ext,
                dpi=300 if ext == "png" else None,
                bbox_inches="tight",
                pad_inches=0.03
            )
        log_message(f"[CHART SAVED] Saved volatility-only standalone legend to {output_folder}")

    plt.show()


def _resolve_plot_context(
        df_graph: pd.DataFrame, focus_metric: str, plot_metric: str, tail_direction: str,
        base_mode_str: str, tail_only: bool, distinguish_tp_fp: bool
) -> Dict[str, Any]:
    """Resolves primary column names, reference columns, labels, and title names strictly from configurations."""
    if tail_direction == "lower":
        tail_ref_col = "Realized_Lower_Prob_%" if "Realized_Lower_Prob_%" in df_graph.columns else "Realized_Tail_Prob_%"
        main_y_col = "False_Trigger_%" if "False_Trigger_%" in df_graph.columns else "Violation_Prob_%"
        rel_y_col = "False_Trigger_Ratio" if "False_Trigger_Ratio" in df_graph.columns else "Relative_Trigger_Ratio"
        rel_label_key = "y_axis_false_trigger_ratio"
        rel_label_default = LABELS_CFG["y_axis_false_trigger_ratio"]
        title_metric_name = LABELS_CFG["title_false_trigger"]
    else:
        tail_ref_col = "Realized_Upper_Prob_%" if "Realized_Upper_Prob_%" in df_graph.columns else "Realized_Tail_Prob_%"
        main_y_col = "Trigger_Miss_%" if "Trigger_Miss_%" in df_graph.columns else "Violation_Prob_%"
        rel_y_col = "Miss_Ratio" if "Miss_Ratio" in df_graph.columns else "Relative_Trigger_Ratio"
        rel_label_key = "y_axis_miss_ratio"
        rel_label_default = LABELS_CFG["y_axis_miss_ratio"]
        title_metric_name = LABELS_CFG["title_trigger_miss"]

    if focus_metric != "trigger_failure" or plot_metric != "probability":
        main_y_col = METRIC_CONFIGS[plot_metric]["col"]
        rel_y_col = "Relative_Trigger_Ratio"
        rel_label_key = "y_axis_relative_ratio"
        rel_label_default = LABELS_CFG["y_axis_relative_ratio"]
        title_metric_name = METRIC_CONFIGS[plot_metric]["title"]

    if tail_only:
        main_y_label = LABELS_CFG["y_axis_tail_lower"] if tail_direction == "lower" else LABELS_CFG["y_axis_tail_upper"]
    elif distinguish_tp_fp and plot_metric == "probability":
        main_y_label = LABELS_CFG["y_axis_signal_breakdown"].format(base_mode_str=base_mode_str)
    elif focus_metric == "trigger_failure" and plot_metric == "probability":
        y_template = LABELS_CFG["y_axis_false_trigger"] if tail_direction == "lower" else LABELS_CFG[
            "y_axis_trigger_miss"]
        main_y_label = y_template.format(base_mode_str=base_mode_str)
    else:
        main_y_label = METRIC_CONFIGS[plot_metric]["label"].format(base_mode_str=base_mode_str)

    rel_raw_template = LABELS_CFG.get(rel_label_key, rel_label_default)
    rel_y_label = rel_raw_template.format(base_mode_str=base_mode_str)

    return {
        "main_y_col": main_y_col,
        "rel_y_col": rel_y_col,
        "tail_ref_col": tail_ref_col,
        "rel_y_label": rel_y_label,
        "main_y_label": main_y_label,
        "title_metric_name": title_metric_name,
    }


def _render_relative_ratio_panel(
        ax_rel: plt.Axes, df_graph: pd.DataFrame, vols: List[float], x_col: str,
        rel_y_col: str, x_min: float, x_max: float, cfg: dict, show_parity: bool,
        rel_y_label: str, linewidth: float = 2.2, markersize: float = 7.0,
        master_pos_vols: Optional[List[float]] = None
) -> None:
    """Renders the top panel using persistent volatility-to-style mapping."""
    rel_axis_label_size = FONTS_CFG["relative_axis_label_size"]
    rel_tick_label_size = FONTS_CFG["relative_tick_label_size"]
    parity_label_size = FONTS_CFG["parity_label_size"]

    is_percentage = rel_y_col in ["Miss_Ratio", "False_Trigger_Ratio"]

    all_y_values = []
    for v in vols:
        df_v = df_graph[df_graph["Monthly_Vol_%"] == v].sort_values(x_col)
        c, m = get_volatility_style(v, master_pos_vols)
        if rel_y_col in df_v.columns:
            y_values = df_v[rel_y_col] * 100.0 if is_percentage else df_v[rel_y_col]
            valid_y = y_values.dropna().values
            if len(valid_y) > 0:
                all_y_values.extend(valid_y)
            ax_rel.plot(
                df_v[x_col], y_values,
                color=c,
                marker=m,
                markevery=max(1, len(df_v) // 25), linewidth=linewidth, markersize=markersize
            )

    peak_y = max(all_y_values) if len(all_y_values) > 0 else 1.0
    top_limit = peak_y * 1.12 if peak_y > 0 else 1.0

    if is_percentage and top_limit > 100.0:
        top_limit = 100.05

    bottom_limit = -0.02 * top_limit if is_percentage else (-0.02 * top_limit if top_limit > 0.1 else -0.004)

    if show_parity and not is_percentage:
        ax_rel.axhline(1.0, color="black", linestyle=":", linewidth=1.8, alpha=0.7, zorder=2)
        parity_str = LABELS_CFG["parity_text"]
        ax_rel.text(x_min + 0.02 * (x_max - x_min), 1.03, parity_str, color="black", fontsize=parity_label_size,
                    fontweight="bold", alpha=0.75)

    ax_rel.set_ylabel(rel_y_label, fontsize=rel_axis_label_size)
    ax_rel.tick_params(axis="both", which="major", labelsize=rel_tick_label_size)
    ax_rel.set_xlim(x_min, x_max)
    ax_rel.set_ylim(bottom=bottom_limit, top=top_limit)
    _apply_grid_and_ticks(ax_rel, cfg)


def _render_main_panel_curves(
        ax_main: plt.Axes, df_graph: pd.DataFrame, vols: List[float], x_col: str,
        main_y_col: str, tail_ref_col: str, plot_metric: str, tail_only: bool,
        distinguish_tp_fp: bool, show_realized_tail: bool, tail_line_cfg: dict,
        linewidth: float = 2.5, markersize: float = 9.0,
        master_pos_vols: Optional[List[float]] = None
) -> None:
    """Renders curves on the primary main panel, dynamically pulling line width and marker size."""
    tail_linestyle = tail_line_cfg["linestyle"]
    tail_linewidth = float(tail_line_cfg["linewidth"])
    tail_alpha = float(tail_line_cfg["alpha"])
    tail_custom_color = tail_line_cfg.get("color", None)

    for v in vols:
        df_v = df_graph[df_graph["Monthly_Vol_%"] == v].sort_values(x_col)
        c, m_style = get_volatility_style(v, master_pos_vols)
        m_step = max(1, len(df_v) // 25)

        if tail_only:
            if tail_ref_col in df_v.columns and v > 0.0:
                ax_main.plot(
                    df_v[x_col], df_v[tail_ref_col],
                    color=tail_custom_color or c, marker=m_style, markevery=m_step,
                    linestyle=tail_linestyle, linewidth=tail_linewidth, alpha=tail_alpha,
                    markersize=markersize
                )
        elif distinguish_tp_fp and plot_metric == "probability":
            if "True_Positive_%" in df_v.columns:
                ax_main.plot(df_v[x_col], df_v["True_Positive_%"], color=c, marker=m_style, markevery=m_step,
                             linewidth=linewidth, markersize=markersize)
            if "False_Trigger_%" in df_v.columns:
                ax_main.plot(df_v[x_col], df_v["False_Trigger_%"], color=c, linestyle="--",
                             linewidth=max(1.0, linewidth - 0.5), alpha=0.85)
            if "Trigger_Miss_%" in df_v.columns:
                ax_main.plot(df_v[x_col], df_v["Trigger_Miss_%"], color=c, linestyle=":",
                             linewidth=max(1.0, linewidth - 0.5), alpha=0.85)
        else:
            if main_y_col in df_v.columns:
                ax_main.plot(df_v[x_col], df_v[main_y_col], color=c, marker=m_style, markevery=m_step,
                             linewidth=linewidth, markersize=markersize)
            if show_realized_tail and plot_metric == "probability" and tail_ref_col in df_v.columns and v > 0.0:
                ax_main.plot(
                    df_v[x_col], df_v[tail_ref_col],
                    color=tail_custom_color or c, linestyle=tail_linestyle,
                    linewidth=tail_linewidth, alpha=tail_alpha
                )


def _format_main_axes(ax_main: plt.Axes, df_graph: pd.DataFrame, cfg: dict, plot_metric: str, main_y_label: str,
                      tail_only: bool, show_realized_tail: bool = True) -> None:
    """Formats labels, limits, and ticks of the main panel, scaling to visible data with headroom padding."""
    axis_label_size = FONTS_CFG["axis_label_size"]
    tick_label_size = FONTS_CFG["tick_label_size"]

    ax_main.tick_params(axis="both", which="major", labelsize=tick_label_size)
    ax_main.set_xlabel(LABELS_CFG["x_axis_theory"], fontsize=axis_label_size)
    ax_main.set_ylabel(main_y_label, fontsize=axis_label_size)
    ax_main.set_xlim(cfg["x_min"], cfg["x_max"])

    if tail_only:
        ax_main.set_ylim(bottom=0, top=100)
        ax_main.set_yticks(np.arange(0, 101, 10))
    elif "count" in plot_metric:
        y_max = max(1.0, np.ceil(df_graph[METRIC_CONFIGS[plot_metric]["col"]].max() * 1.15))
        ax_main.set_ylim(bottom=0, top=y_max)
        ax_main.locator_params(axis="y", nbins=8)
    else:
        candidates = [df_graph["Violation_Prob_%"].dropna().max()]
        eval_cols = ["True_Positive_%", "False_Positive_%", "False_Negative_%", "Trigger_Miss_%", "False_Trigger_%"]

        if show_realized_tail:
            eval_cols.extend(["Realized_Upper_Prob_%", "Realized_Lower_Prob_%"])

        for col in eval_cols:
            if col in df_graph.columns:
                val = df_graph[col].dropna().max()
                if pd.notna(val):
                    candidates.append(val)

        highest = max(candidates) if candidates else 10.0
        padded_highest = highest * 1.12

        if padded_highest <= 20:
            y_max_tick = max(5, int(np.ceil(padded_highest / 5.0) * 5))
            step = 5
        elif padded_highest <= 50:
            y_max_tick = int(np.ceil(padded_highest / 10.0) * 10)
            step = 10
        else:
            y_max_tick = int(np.ceil(padded_highest / 20.0) * 20)
            step = 20

        if y_max_tick > 100:
            y_max_tick = 100
            step = 20

        top_limit = 100.05 if y_max_tick == 100 else float(y_max_tick)
        ax_main.set_ylim(bottom=-0.1, top=top_limit)
        ax_main.set_yticks(np.arange(0, y_max_tick + 1, step))

    _apply_grid_and_ticks(ax_main, cfg)


def _draw_vertical_drift_parity(target_axes: List[plt.Axes], x_min: float, x_max: float) -> None:
    """Draws the unified vertical Constant Drift Line (X = 1.0) across all active axes."""
    toggles = CONFIG["plotting"]["toggles"]
    drift_cfg = CONFIG["plotting"]["drift_parity_line"]
    show_x_drift = toggles["show_drift_parity"] if "show_drift_parity" in toggles else CONFIG["simulation"][
        "show_drift_parity"]
    x_drift_val = drift_cfg["x"]

    if show_x_drift and (x_min <= x_drift_val <= x_max):
        for target_ax in target_axes:
            target_ax.axvline(
                x=x_drift_val,
                color=drift_cfg["color"],
                linestyle=drift_cfg["linestyle"],
                linewidth=float(drift_cfg["linewidth"]),
                alpha=float(drift_cfg["alpha"]),
                zorder=999,
                clip_on=False
            )


def _attach_title_and_legend(
        title_ax: plt.Axes, legend_ax: plt.Axes, cfg: dict, handles: List, labels: List,
        title_metric_name: str, base_mode_str: str, eval_frequency: str, sim_proc: int,
        tail_direction: str, tail_only: bool
) -> None:
    """Formats and places title and legend strictly from TOML without silent defaults."""
    proc_str = "LOGNORMAL" if sim_proc == 1 else "AR-GARCH"

    if tail_only:
        formatted_title = LABELS_CFG["title_tail_only"]
    else:
        title_tpl = LABELS_CFG["title_template"]
        formatted_title = title_tpl.format(
            title=title_metric_name, base_mode_str=base_mode_str,
            eval_freq=eval_frequency.upper(), proc_str=proc_str,
            tail_dir=tail_direction.upper(), m=cfg["m"], p1=cfg["p1"], p2=cfg["p2"]
        )

    title_size = FONTS_CFG["title_size"]
    title_pad = FONTS_CFG["title_pad"]
    leg_title_raw = LABELS_CFG["legend_vol_title"]
    leg_title_size = FONTS_CFG["legend_title_size"]
    leg_font_size = FONTS_CFG["legend_font_size"]

    title_ax.set_title(formatted_title, fontsize=title_size, pad=title_pad)

    legend_ax.legend(
        handles, labels, title=leg_title_raw,
        title_fontproperties={"weight": "bold", "size": leg_title_size}, fontsize=leg_font_size,
        loc=cfg["legend_loc"], markerscale=1.5, borderpad=0.3, labelspacing=0.35
    )


def _save_plot_files(output_folder: str, cfg: dict, plot_metric: str, eval_frequency: str, sim_proc: int,
                     tail_direction: str, tail_only: bool, is_stacked: bool, distinguish_tp_fp: bool, focus_metric: str,
                     fig: plt.Figure) -> None:
    """Saves SVG and PNG exports."""
    toggles = CONFIG["plotting"]["toggles"]
    if not toggles.get("save_fig", True):
        return

    mode, m, p1, p2, N_eval = cfg["mode"], cfg["m"], cfg["p1"], cfg["p2"], cfg["n_eval_months"]
    x_min, x_max = cfg["x_min"], cfg["x_max"]
    sample_tag = f"sampl_{x_min}_{x_max}"
    prefix = "red" if mode == "reduction" else "incr"
    proc_tag = "_lognorm" if sim_proc == 1 else "_garch"
    rel_tag = "_with_relative" if is_stacked else ""
    tail_only_tag = "_tail_only" if tail_only else ""
    tp_tag = "_tp_fp_fn" if (distinguish_tp_fp and not tail_only) else (
        f"_{focus_metric}_{tail_direction}" if (focus_metric == "trigger_failure" and not tail_only) else "")
    legend_suffix = "_no_legend" if toggles.get("without_legend", False) else ""

    metric_tag = METRIC_CONFIGS.get(plot_metric, {})["tag"]
    base_name = f"{output_folder}{prefix}{metric_tag}_curves_daily_sim_m{m}_p1{p1}_p2{p2}_months_{N_eval}_{sample_tag}_{eval_frequency}{proc_tag}_theory_{tail_direction}{tp_tag}{rel_tag}{tail_only_tag}{legend_suffix}"
    fig.savefig(f"{base_name}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{base_name}.png", format="png", dpi=300, bbox_inches="tight")
    log_message(f"[CHART SAVED] Saved plot to: {base_name}.svg & .png")


# ==============================================================================
# ORCHESTRATOR PLOTTING FUNCTION
# ==============================================================================
def plot_theoretical_drift_curves(
        df_graph: pd.DataFrame, cfg: dict, output_folder: str, plot_metric: str,
        eval_frequency: str = "monthly", sim_proc: int = 0, relative_metric: bool = False,
        tail_direction: str = "upper", distinguish_tp_fp: bool = False,
        focus_metric: str = "trigger", show_realized_tail: bool = True
) -> None:
    """Orchestrates figure creation, panel rendering, formatting, and file export."""
    os.makedirs(output_folder, exist_ok=True)
    toggles = CONFIG["plotting"]["toggles"]
    tail_line_cfg = CONFIG["plotting"]["tail_reference_line"]
    tail_only = tail_line_cfg["only"] if "only" in tail_line_cfg else toggles["only_tail_reference"]
    show_parity = toggles["show_parity"] if "show_parity" in toggles else CONFIG["simulation"]["show_parity"]

    base_mode_str = f"m=1/{cfg['m']}" if cfg["mode"] == "reduction" else f"m={cfg['m']}"

    # 1. Build persistent list of positive volatilities for fixed color/marker mapping
    raw_vols = sorted(df_graph["Monthly_Vol_%"].unique())
    cfg_pos_vols = [float(v) for v in CONFIG["simulation"].get("volatility_list", []) if float(v) > 0.0]
    data_pos_vols = [v for v in raw_vols if v > 0.0]
    master_pos_vols = sorted(list(set(cfg_pos_vols + data_pos_vols)))

    # When tail_only is enabled, exclude zero volatility completely from the curves and legend
    if tail_only:
        vols = [v for v in raw_vols if v > 0.0]
    else:
        vols = raw_vols

    x_col = "Target_Yearly_X"
    x_min, x_max = cfg["x_min"], cfg["x_max"]

    # 2. Resolve Columns, Ratios, and Labels
    ctx = _resolve_plot_context(df_graph, focus_metric, plot_metric, tail_direction, base_mode_str, tail_only,
                                distinguish_tp_fp)

    # 3. Figure Layout Setup
    is_stacked = relative_metric and (ctx["rel_y_col"] in df_graph.columns) and (not tail_only)
    if is_stacked:
        fig, (ax_rel, ax_main) = plt.subplots(2, 1, figsize=(11, 9.5), dpi=150, sharex=True,
                                              # gridspec_kw={"height_ratios": [1, 1.25]})
                                              gridspec_kw={"height_ratios": [1, 1]})
    else:
        fig, ax_main = plt.subplots(figsize=(11, 6.5), dpi=150)
        ax_rel = None

    (
        handles, labels,
        vol_handles, vol_labels, vol_title,
        style_handles, style_labels, style_title
    ) = _build_legend_artists(
        vols, tail_direction=tail_direction, distinguish_tp_fp=distinguish_tp_fp,
        focus_metric=focus_metric, tail_only=tail_only,
        master_pos_vols=master_pos_vols
    )

    if toggles["only_legend"]:
        sample_tag = f"sampl_{x_min}_{x_max}"
        prefix = "red" if cfg["mode"] == "reduction" else "incr"
        proc_tag = "_lognorm" if sim_proc == 1 else "_garch"
        rel_tag = "_with_relative" if is_stacked else ""
        tail_only_tag = "_tail_only" if tail_only else ""
        tp_tag = "_tp_fp_fn" if (distinguish_tp_fp and not tail_only) else (
            f"_{focus_metric}_{tail_direction}" if (focus_metric == "trigger_failure" and not tail_only) else "")

        _save_standalone_legend(
            fig_main=fig,
            vol_handles=vol_handles,
            vol_labels=vol_labels,
            vol_title=vol_title,
            save_prefix=f"{prefix}{proc_tag}_theory_{tail_direction}{tp_tag}{rel_tag}{tail_only_tag}",
            sample_tag=sample_tag,
            output_folder=output_folder,
            save_fig=toggles.get("save_fig", True)
        )
        return

    toml_linewidth = float(cfg.get("linewidth", 2.5))
    toml_markersize = float(cfg.get("markersize", 9.0))

    # 4. Render Panels
    if ax_rel is not None:
        ratio_linewidth = max(1.0, toml_linewidth - 0.3)
        ratio_markersize = max(1.0, toml_markersize - 2.0)
        _render_relative_ratio_panel(
            ax_rel=ax_rel,
            df_graph=df_graph,
            vols=vols,
            x_col=x_col,
            rel_y_col=ctx["rel_y_col"],
            x_min=x_min,
            x_max=x_max,
            cfg=cfg,
            show_parity=show_parity,
            rel_y_label=ctx["rel_y_label"],
            linewidth=ratio_linewidth,
            markersize=ratio_markersize,
            master_pos_vols=master_pos_vols
        )

    _render_main_panel_curves(
        ax_main, df_graph, vols, x_col, ctx["main_y_col"], ctx["tail_ref_col"],
        plot_metric, tail_only, distinguish_tp_fp, show_realized_tail,
        tail_line_cfg, linewidth=toml_linewidth, markersize=toml_markersize,
        master_pos_vols=master_pos_vols
    )
    _format_main_axes(ax_main, df_graph, cfg, plot_metric, ctx["main_y_label"], tail_only, show_realized_tail)

    # 5. Overlays, Typography & File Export
    _draw_vertical_drift_parity([ax_rel, ax_main] if ax_rel is not None else [ax_main], x_min, x_max)

    if not toggles["without_legend"]:
        _attach_title_and_legend(ax_rel or ax_main, ax_main, cfg, handles, labels, ctx["title_metric_name"],
                                 base_mode_str, eval_frequency, sim_proc, tail_direction, tail_only)

    plt.tight_layout()
    _save_plot_files(output_folder, cfg, plot_metric, eval_frequency, sim_proc, tail_direction, tail_only, is_stacked,
                     distinguish_tp_fp, focus_metric, fig)
    plt.show()


# ==============================================================================
# CONFIGURATION BUILDER & EXPERIMENT DISPATCHER
# ==============================================================================
def build_run_configs(config: dict, mode: str, tail_direction: str) -> List[dict]:
    """Extracts run configurations supporting 3-level nesting ([runs.<tail>.<mode>]) and boundary defaults."""
    base = {**config["stochastic"], **config["simulation"]}
    sim_cfg = config.get("simulation", {})
    configs = []
    raw_runs = config.get("runs", [])

    default_markersize = float(config["plotting"]["markersize"])

    if tail_direction == "lower":
        default_x_min = float(sim_cfg["default_x_lower_min"])
        default_x_max = float(sim_cfg["default_x_lower_max"])
    else:
        default_x_min = float(sim_cfg["default_x_upper_min"])
        default_x_max = float(sim_cfg["default_x_upper_max"])

    if isinstance(raw_runs, list):
        for r in raw_runs:
            shared = {k: v for k, v in r.items() if k not in ["upper", "lower", "increase", "reduction"]}
            mode_sub = r.get(tail_direction, {}).get(mode, r.get(mode, r.get(tail_direction, {})))

            if isinstance(mode_sub, dict) and mode_sub:
                merged = {**base, **shared, **mode_sub, "mode": mode, "tail_direction": tail_direction}
                merged["x_min"] = float(mode_sub.get("x_min", shared.get("x_min", default_x_min)))
                merged["x_max"] = float(mode_sub.get("x_max", shared.get("x_max", default_x_max)))
                merged["linewidth"] = float(mode_sub.get("linewidth", shared.get("linewidth", 2.5)))
                merged["markersize"] = float(mode_sub.get("markersize", shared.get("markersize", default_markersize)))
                configs.append(merged)
            elif not any(k in r for k in ["upper", "lower", "increase", "reduction"]):
                merged = {**base, **r, "mode": mode, "tail_direction": tail_direction}
                merged["x_min"] = float(r.get("x_min", default_x_min))
                merged["x_max"] = float(r.get("x_max", default_x_max))
                merged["linewidth"] = float(r.get("linewidth", 2.5))
                merged["markersize"] = float(r.get("markersize", default_markersize))
                configs.append(merged)

    elif isinstance(raw_runs, dict):
        for r in raw_runs.get(mode, raw_runs.get(tail_direction, [])):
            merged = {**base, **r, "mode": mode, "tail_direction": tail_direction}
            merged["x_min"] = float(r.get("x_min", default_x_min))
            merged["x_max"] = float(r.get("x_max", default_x_max))
            merged["linewidth"] = float(r.get("linewidth", 2.5))
            merged["markersize"] = float(r.get("markersize", default_markersize))
            configs.append(merged)

    return configs


def execute_experiment_suite(runs_list: List[dict], suite_name: str, njobs: int, tail_direction: str) -> None:
    """Iterates through runs, executes simulations, updates database, and plots."""
    sim = CONFIG["simulation"]
    toggles = CONFIG["plotting"]["toggles"]

    eval_freq = sim["eval_frequency"]
    sim_proc = sim["simulation_process"]
    relative_metric = toggles["relative_metric"] if "relative_metric" in toggles else sim["relative_metric"]
    show_count = toggles["show_count"] if "show_count" in toggles else sim["show_count"]
    distinguish_tp_fp = toggles[
        "distinguish_true_from_false_positives"] if "distinguish_true_from_false_positives" in toggles else sim[
        "distinguish_true_from_false_positives"]
    focus_metric = sim["focus_metric"]
    show_realized_tail = sim["show_realized_tail"]

    tail_line_cfg = CONFIG["plotting"]["tail_reference_line"]
    tail_only = tail_line_cfg["only"] if "only" in tail_line_cfg else toggles["only_tail_reference"]

    proc_status = "CONSTANT VOLATILITY LOGNORMAL (GBM)" if sim_proc == 1 else "AR(1)-GARCH(1,1)"
    grid_status = f"THEORETICAL DRIFT ({tail_direction.upper()} TAIL)"

    if tail_only:
        metrics_to_run = ["probability"]
        runs_list = runs_list[:1]
    elif not show_count:
        metrics_to_run = [m for m in sim["metrics_to_plot"] if "count" not in m]
    else:
        metrics_to_run = sim["metrics_to_plot"]

    log_message("=" * 80)
    log_message(
        f"=== STARTING {suite_name.upper()} EXPERIMENTS [{proc_status} | {grid_status} | EVAL: {eval_freq.upper()}] ===")
    log_message("=" * 80)

    for cfg in runs_list:
        p1, p2, m, mode = cfg["p1"], cfg["p2"], cfg["m"], cfg["mode"]

        if cfg.get("warm_up_max", False) and not tail_only:
            diag = log_warmup_diagnostics(p1=p1, p2=p2, m=m, mode=mode, S0=100.0)
            log_message(f"\n--- [WARM-UP MAX DIAGNOSTICS: m={m}, p1={p1}, p2={p2}, mode={mode.upper()}] ---")
            log_message(
                f"  * Solved Growth Factor (r): {diag['r_monthly']:.8f} | X_crit: {diag['r_annual_X_crit']:.4f}")
            log_message(
                f"  * Resulting MA Ratio:       {diag['eval_ratio']:.8f} (Discrepancy: {diag['discrepancy']:.2e})\n")

        df_graph = generate_theoretical_drift_data_cached(
            cfg=cfg, vol_list=sim["volatility_list"], phi_daily=cfg["phi_daily"],
            n_sims=sim["n_sims"], batch_size=sim["batch_size"],
            n_eval_months=sim["n_eval_months"], n_jobs=njobs,
            use_cache=sim["use_cache"], cache_dir=sim["cache_dir"],
            eval_frequency=eval_freq, sim_proc=sim_proc, tail_direction=tail_direction
        )

        if not tail_only:
            _update_master_database(df_graph, cfg, sim_proc, tail_direction)

        for metric in metrics_to_run:
            plot_theoretical_drift_curves(
                df_graph=df_graph, cfg=cfg,
                output_folder=CONFIG["paths"]["output_figures"],
                plot_metric=metric,
                eval_frequency=eval_freq,
                sim_proc=sim_proc,
                relative_metric=relative_metric,
                tail_direction=tail_direction,
                distinguish_tp_fp=distinguish_tp_fp,
                focus_metric=focus_metric,
                show_realized_tail=show_realized_tail
            )


# # ==============================================================================
# # MAIN ENTRYPOINT
# # ==============================================================================
# if __name__ == "__main__":
#     os.makedirs(CONFIG["paths"]["output_figures"], exist_ok=True)
#     os.makedirs(CONFIG["paths"]["output_data_dir"], exist_ok=True)
#
#     cpu_count = multiprocessing.cpu_count()
#     sim = CONFIG["simulation"]
#     toggles = CONFIG["plotting"]["toggles"]
#
#     default_safe_jobs = max(1, cpu_count // 2)
#     njobs = min(sim.get("max_workers", default_safe_jobs), cpu_count - 2)
#
#     eval_freq = sim["eval_frequency"]
#     sim_proc = sim["simulation_process"]
#     tail_dir = sim["tail_direction"].lower()
#     if tail_dir not in ["upper", "lower"]:
#         tail_dir = "upper"
#
#     tail_line_cfg = CONFIG["plotting"]["tail_reference_line"]
#     tail_only = tail_line_cfg["only"] if "only" in tail_line_cfg else toggles["only_tail_reference"]
#
#     log_message(
#         f"[PIPELINE START] Program=THEORETICAL DRIFT, tail_direction={tail_dir.upper()}, tail_only={tail_only}, n_sims={sim['n_sims']:,}, eval_frequency={eval_freq}, workers={njobs}"
#     )
#
#     runs_increase = build_run_configs(CONFIG, "increase", tail_dir)
#     runs_reduction = build_run_configs(CONFIG, "reduction", tail_dir)
#
#     if tail_only:
#         log_message(
#             "[INFO] [plotting.tail_reference_line] only = true -> Generating exactly ONE baseline reference figure.")
#         active_run = runs_increase[:1] if len(runs_increase) > 0 else runs_reduction[:1]
#         if len(active_run) > 0:
#             execute_experiment_suite(active_run, f"BASELINE {tail_dir.upper()} TAIL DRIFT DISTRIBUTION", njobs,
#                                      tail_dir)
#         log_message("[PIPELINE COMPLETE] Baseline reference figure generated successfully.")
#         sys.exit(0)
#
#     if toggles["only_legend"]:
#         runs_increase = runs_increase[:1]
#         runs_reduction = runs_reduction[:1]
#
#     if sim["include_increase"] and len(runs_increase) > 0:
#         execute_experiment_suite(runs_increase, "PRICE INCREASE (> m)", njobs, tail_dir)
#     elif not sim["include_increase"]:
#         log_message("[SKIP] Increase experiments skipped (include_increase = false).")
#     else:
#         log_message("[WARNING] No runs found for mode='increase'.")
#
#     if sim["include_reduction"] and len(runs_reduction) > 0:
#         execute_experiment_suite(runs_reduction, "PRICE REDUCTION (< 1/m)", njobs, tail_dir)
#     elif not sim["include_reduction"]:
#         log_message("[SKIP] Reduction experiments skipped (include_reduction = false).")
#     else:
#         log_message("[WARNING] No runs found for mode='reduction'.")
#
#     log_message("[PIPELINE COMPLETE] All theoretical drift experiments completed successfully!")