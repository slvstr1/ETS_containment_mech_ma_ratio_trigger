# generate_simulations.py
import os
import sys
import datetime
import hashlib
import json
import multiprocessing
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Set single-thread environment variables before importing numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from utils import load_config
from simulation_engine import (
    solve_max_growth_r,
    log_warmup_diagnostics,
)
from plot_engine import (
    get_critical_X,
    eval_theory_grid_point_streaming,
    build_run_configs
)

CONFIG = load_config()


def generate_and_save_data(cfg: dict, sim_proc: int, tail_direction: str, njobs: int, output_dir: str) -> None:
    """Runs simulations for a specific configuration and persists data + metadata."""
    os.makedirs(output_dir, exist_ok=True)

    p1, p2, m, mode = cfg["p1"], cfg["p2"], cfg["m"], cfg["mode"]
    x_min, x_max = cfg["x_min"], cfg["x_max"]
    num_x_points = cfg.get("num_x_points", 25)
    warm_up_max = cfg.get("warm_up_max", False)
    eval_freq = CONFIG["simulation"]["eval_frequency"]
    dpm = cfg["days_per_month"]
    n_sims = CONFIG["simulation"]["n_sims"]
    batch_size = CONFIG["simulation"]["batch_size"]
    n_eval_months = CONFIG["simulation"]["n_eval_months"]
    vol_list = CONFIG["simulation"]["volatility_list"]
    phi_daily = cfg["phi_daily"]

    eff_phi = 0.0 if sim_proc == 1 else phi_daily
    eff_alpha = 0.0 if sim_proc == 1 else cfg["alpha"]
    eff_beta = 0.0 if sim_proc == 1 else cfg["beta"]

    # 1. Warm-up Diagnostics
    if warm_up_max:
        diag = log_warmup_diagnostics(p1=p1, p2=p2, m=m, mode=mode, S0=100.0)
        print(f"[WARM-UP] Growth Factor (r): {diag['r_monthly']:.8f} | X_crit: {diag['r_annual_X_crit']:.4f}")

    # 2. Build target yearly and cumulative arrays
    x_yearly_targets = np.linspace(x_min, x_max, num_x_points)
    x_crit = get_critical_X(p1, p2, m, mode, eval_freq, dpm)
    if x_min <= x_crit <= x_max:
        micro_offsets = np.array([x_crit - 1e-5, x_crit, x_crit + 1e-5])
        x_yearly_targets = np.unique(np.sort(np.concatenate([x_yearly_targets, micro_offsets])))

    years = n_eval_months / 12.0
    x_cumulative_targets = x_yearly_targets ** years

    # 3. Build Parallel Tasks
    tasks = [
        (
            x_val, x_cum, v, p1, p2, m, n_sims, batch_size,
            n_eval_months, eff_phi, eff_alpha, eff_beta, dpm,
            eval_freq, warm_up_max, sim_proc, tail_direction, mode
        )
        for v in vol_list for x_val, x_cum in zip(x_yearly_targets, x_cumulative_targets)
    ]

    print(f"[RUN START] p1={p1}, p2={p2}, m={m}, proc={sim_proc}, tail={tail_direction}, mode={mode}")
    results = Parallel(n_jobs=njobs, batch_size=4)(
        delayed(eval_theory_grid_point_streaming)(t) for t in tasks
    )

    df_out = pd.DataFrame(results)

    # 4. Save file outputs
    file_base = f"run_p1_{p1}_p2_{p2}_m_{m}_proc{sim_proc}_tail_{tail_direction}_mode_{mode}"
    data_path = os.path.join(output_dir, f"{file_base}_data.parquet")
    metadata_path = os.path.join(output_dir, f"{file_base}_metadata.json")

    df_out.to_parquet(data_path, index=False)

    metadata = {
        "p1": int(p1),
        "p2": int(p2),
        "m": float(m),
        "mode": mode,
        "tail_direction": tail_direction,
        "simulation_process": int(sim_proc),
        "eval_frequency": eval_freq,
        "days_per_month": int(dpm),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "n_sims": int(n_sims),
        "n_eval_months": int(n_eval_months),
        "volatility_list": [float(v) for v in vol_list],
        "warm_up_max": bool(warm_up_max),
        "phi_daily": float(phi_daily),
        "alpha": float(cfg["alpha"]),
        "beta": float(cfg["beta"]),
        "linewidth": float(cfg.get("linewidth", 2.5)),
        "markersize": float(cfg.get("markersize", 9.0)),
        "legend_loc": cfg.get("legend_loc", "upper left"),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[COMPLETED & SAVED] -> {data_path}\n")


if __name__ == "__main__":
    output_directory = "simulation_results"
    cpu_count = multiprocessing.cpu_count()
    sim_settings = CONFIG["simulation"]

    default_safe_jobs = max(1, cpu_count // 2)
    njobs = min(sim_settings.get("max_workers", default_safe_jobs), cpu_count - 2)

    sim_proc = sim_settings["simulation_process"]
    tail_dir = sim_settings["tail_direction"].lower()

    runs_increase = build_run_configs(CONFIG, "increase", tail_dir)
    runs_reduction = build_run_configs(CONFIG, "reduction", tail_dir)

    if sim_settings["include_increase"]:
        for run_cfg in runs_increase:
            generate_and_save_data(run_cfg, sim_proc, tail_dir, njobs, output_directory)

    if sim_settings["include_reduction"]:
        for run_cfg in runs_reduction:
            generate_and_save_data(run_cfg, sim_proc, tail_dir, njobs, output_directory)

    print("All generation tasks are complete.")