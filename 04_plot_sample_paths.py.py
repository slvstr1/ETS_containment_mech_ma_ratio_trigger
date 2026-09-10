import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import load_config
from simulation_engine import simulate_ar_garch_daily_batch, get_daily_c_for_target_X

CONFIG = load_config()


def plot_synthetic_vs_historical(
        df_monthly: pd.DataFrame,
        simulated_paths: np.ndarray,
        pv_cfg: dict,
        price_cfg: dict,
        fonts_cfg: dict,
        fig_dir: str,
        start_year: int,
        save_fig: bool = True
):
    os.makedirs(fig_dir, exist_ok=True)
    axis_label_size = fonts_cfg.get("axis_label_size", 18)
    tick_label_size = fonts_cfg.get("tick_label_size", 14)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    timeline = df_monthly.index

    # 1. Shaded Confidence Fan
    if pv_cfg.get("show_confidence_fan", True):
        p10 = np.percentile(simulated_paths, 10, axis=1)
        p25 = np.percentile(simulated_paths, 25, axis=1)
        p75 = np.percentile(simulated_paths, 75, axis=1)
        p90 = np.percentile(simulated_paths, 90, axis=1)

        ax.fill_between(
            timeline, p10, p90,
            color=pv_cfg.get("fan_color", "lightsteelblue"),
            alpha=pv_cfg.get("fan_alpha", 0.35),
            label="Simulated 10%–90% Confidence Interval",
            zorder=1
        )
        ax.fill_between(
            timeline, p25, p75,
            color=pv_cfg.get("fan_color", "lightsteelblue"),
            alpha=pv_cfg.get("fan_alpha", 0.35) * 1.5,
            label="Simulated 25%–75% Interquartile Range",
            zorder=2
        )

    # 2. Individual Sample Trajectories
    if pv_cfg.get("show_individual_paths", True):
        n_show = min(pv_cfg.get("n_sample_paths", 15), simulated_paths.shape[1])
        for i in range(n_show):
            lbl = "Sample Simulated Paths" if i == 0 else None
            ax.plot(
                timeline, simulated_paths[:, i],
                color=pv_cfg.get("path_color", "steelblue"),
                alpha=pv_cfg.get("path_alpha", 0.4),
                linewidth=pv_cfg.get("path_linewidth", 1.0),
                label=lbl,
                zorder=3
            )

    # 3. Simulated Median
    if pv_cfg.get("show_median", True):
        median_path = np.median(simulated_paths, axis=1)
        ax.plot(
            timeline, median_path,
            color=pv_cfg.get("median_color", "navy"),
            linestyle=pv_cfg.get("median_linestyle", "--"),
            linewidth=pv_cfg.get("median_linewidth", 2.0),
            label="Simulated Median Path",
            zorder=4
        )

    # 4. Actual Historical Price
    price_color = price_cfg.get("color", "red")
    ax.plot(
        timeline, df_monthly["price"],
        color=price_color,
        linestyle=price_cfg.get("linestyle", "-"),
        linewidth=price_cfg.get("linewidth", 3.0),
        label=f"Historical Price (Actual, from {start_year})",
        zorder=5
    )

    ax.set_xlabel("Date (Months)", fontsize=axis_label_size)
    ax.set_ylabel("EUA Price (€)", fontsize=axis_label_size, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    plt.title(f"Synthetic AR-GARCH Paths vs. Historical Realized EUA Prices (from {start_year})", fontsize=15, pad=15)
    plt.tight_layout()

    if save_fig:
        save_path = os.path.join(fig_dir, f"synthetic_vs_historical_paths_{start_year}.svg")
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        print(f"[SUCCESS] Saved path comparison figure to: {save_path}")

    plt.show()


def main():
    stoch_cfg = CONFIG["stochastic"]
    pv_cfg = CONFIG.get("path_visualization", {})
    toggles = CONFIG["plotting"]["toggles"]
    fonts_cfg = CONFIG["plotting"].get("fonts", {})
    price_cfg = CONFIG["plotting"].get("historical_price", {"color": "red"})

    start_year = CONFIG["data"]["start_year"]
    out_dir = CONFIG["paths"]["output_data_dir"]
    fig_dir = CONFIG["paths"]["output_figures"]

    input_csv = os.path.join(out_dir, f"merged_eua_{start_year}_2026.csv")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Missing {input_csv}. Please run Program 1 first.")

    df_raw = pd.read_csv(input_csv, parse_dates=["date"]).set_index("date")
    df_monthly = df_raw.resample("MS").mean()

    N_months = len(df_monthly)
    S0 = float(df_monthly["price"].iloc[0])

    alpha = stoch_cfg["alpha"]
    beta = stoch_cfg["beta"]
    phi_daily = stoch_cfg["phi_daily"]
    dpm = stoch_cfg["days_per_month"]
    vol_pct = pv_cfg.get("vol_pct", 17.86)

    if pv_cfg.get("drift_mode", "historical") == "target_x":
        target_annual_x = pv_cfg.get("target_yearly_x", 1.0)
        c_daily, sigma_daily = get_daily_c_for_target_X(target_annual_x ** (N_months / 12.0), N_months, vol_pct,
                                                        phi_daily, dpm)
    else:
        realized_total_growth = df_monthly["price"].iloc[-1] / S0
        c_daily, sigma_daily = get_daily_c_for_target_X(realized_total_growth, N_months, vol_pct, phi_daily, dpm)

    omega_daily = (sigma_daily ** 2) * (1.0 - alpha - beta)
    n_sims = pv_cfg.get("n_background_sims", 1000)

    print(f"Simulating {n_sims} AR-GARCH paths starting from S0 = €{S0:.2f} across {N_months} months...")
    simulated_paths = simulate_ar_garch_daily_batch(
        N_months=N_months, batch_size=n_sims, S0=S0, c_daily=c_daily,
        phi_daily=phi_daily, omega_daily=omega_daily, alpha=alpha, beta=beta, days_per_month=dpm
    )

    plot_synthetic_vs_historical(
        df_monthly=df_monthly, simulated_paths=simulated_paths, pv_cfg=pv_cfg,
        price_cfg=price_cfg, fonts_cfg=fonts_cfg, fig_dir=fig_dir,
        start_year=start_year, save_fig=toggles.get("save_fig", True)
    )


if __name__ == "__main__":
    main()