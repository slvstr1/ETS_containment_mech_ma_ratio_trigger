# draw_simulations.py
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from utils import load_config
from plot_engine import plot_theoretical_drift_curves

CONFIG = load_config()


def draw_from_saved_files(results_dir: str, figures_output_dir: str) -> None:
    """Scans the results directory and renders visualizations for each distinct simulation pair."""
    # Find all saved metadata files
    metadata_pattern = os.path.join(results_dir, "*_metadata.json")
    metadata_files = glob.glob(metadata_pattern)

    if not metadata_files:
        print(f"No metadata files found in '{results_dir}'. Run the generator script first.")
        return

    print(f"Found {len(metadata_files)} simulation datasets to process.")

    toggles = CONFIG["plotting"]["toggles"]
    sim_cfg = CONFIG["simulation"]
    show_count = toggles["show_count"] if "show_count" in toggles else sim_cfg["show_count"]

    if not show_count:
        metrics_to_plot = [m for m in sim_cfg["metrics_to_plot"] if "count" not in m]
    else:
        metrics_to_plot = sim_cfg["metrics_to_plot"]

    relative_metric = toggles["relative_metric"] if "relative_metric" in toggles else sim_cfg["relative_metric"]
    distinguish_tp_fp = toggles[
        "distinguish_true_from_false_positives"] if "distinguish_true_from_false_positives" in toggles else sim_cfg[
        "distinguish_true_from_false_positives"]
    focus_metric = sim_cfg["focus_metric"]
    show_realized_tail = sim_cfg["show_realized_tail"]

    for meta_path in metadata_files:
        # Resolve matching parquet data file
        data_path = meta_path.replace("_metadata.json", "_data.parquet")
        if not os.path.exists(data_path):
            print(f"[WARNING] Skipping. Data file does not exist for: {meta_path}")
            continue

        # Load metadata configuration
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_cfg = json.load(f)

        print("=" * 80)
        print(f"LOGGING RUN PARAMETERS:")
        print(f"  * Mode:           {meta_cfg['mode'].upper()}")
        print(f"  * Window Window:  [{meta_cfg['x_min']}, {meta_cfg['x_max']}]")
        print(f"  * Process Type:   {'Lognormal (GBM)' if meta_cfg['simulation_process'] == 1 else 'AR(1)-GARCH(1,1)'}")
        print(f"  * MA window (p1): {meta_cfg['p1']}")
        print(f"  * MA window (p2): {meta_cfg['p2']}")
        print(f"  * MA threshold m: {meta_cfg['m']}")
        print(f"  * Paths Simulated: {meta_cfg['n_sims']:,}")
        print("=" * 80)

        # Load dataset
        df_graph = pd.read_parquet(data_path)

        # Generate plot output directory
        os.makedirs(figures_output_dir, exist_ok=True)

        # Plot specified metrics
        for metric in metrics_to_plot:
            plot_theoretical_drift_curves(
                df_graph=df_graph,
                cfg=meta_cfg,
                output_folder=figures_output_dir,
                plot_metric=metric,
                eval_frequency=meta_cfg["eval_frequency"],
                sim_proc=meta_cfg["simulation_process"],
                relative_metric=relative_metric,
                tail_direction=meta_cfg["tail_direction"],
                distinguish_tp_fp=distinguish_tp_fp,
                focus_metric=focus_metric,
                show_realized_tail=show_realized_tail
            )


if __name__ == "__main__":
    results_input_directory = "simulation_results"
    figures_export_directory = CONFIG["paths"]["output_figures"]

    draw_from_saved_files(results_input_directory, figures_export_directory)
    print("Drawing and export pipeline has completed successfully.")