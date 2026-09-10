import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import our configuration loader
from utils import load_config


# ==============================================================================
# DATA PROCESSING
# ==============================================================================
def load_and_calculate_ratios(filepath: str, configs: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads historical daily price data, resamples to monthly, and calculates MA ratios."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find {filepath}. Please run Program 1 (Estimation) first.")

    # 1. Load Daily Data
    df_daily = pd.read_csv(filepath, parse_dates=["date"])
    df_daily.set_index("date", inplace=True)

    # 2. Resample to Monthly Arithmetic Average
    df_monthly = df_daily.resample("MS").mean()

    # 3. Calculate MA Ratios dynamically based on TOML array
    for cfg in configs:
        p1, p2 = cfg["p1"], cfg["p2"]
        col_name = f"ratio_{p1}_{p2}"

        # Recent p2 months rolling average
        ma_p2 = df_monthly["price"].rolling(window=p2).mean()

        # Preceding p1 months rolling average (shift by p2 so they don't overlap)
        ma_p1 = df_monthly["price"].shift(p2).rolling(window=p1).mean()

        # Calculate Ratio
        df_monthly[col_name] = ma_p2 / ma_p1

    return df_monthly, df_daily


# ==============================================================================
# STANDALONE LEGEND EXPORTER
# ==============================================================================
def _save_standalone_legend(fig_main, handles, labels, save_path: str, save_fig: bool, font_size: int = 10):
    """Exports a tight-cropped image containing ONLY the unified legend."""
    plt.close(fig_main)
    fig_leg, ax_leg = plt.subplots(figsize=(6, 4), dpi=150)
    ax_leg.axis("off")

    leg = ax_leg.legend(
        handles, labels, loc="center", fontsize=font_size,
        frameon=True, framealpha=0.95, borderpad=0.5, labelspacing=0.4
    )

    if save_fig:
        fig_leg.canvas.draw()
        bbox = leg.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
        fig_leg.savefig(save_path, format="svg", bbox_inches=bbox, pad_inches=0.02)
        print(f"[SUCCESS] Saved standalone legend to: {save_path}")

    plt.show()


# ==============================================================================
# PLOTTING ROUTINE
# ==============================================================================
def plot_historical_ratios(
        df_monthly: pd.DataFrame,
        df_daily: pd.DataFrame,
        ma_configs: list,
        thresholds: list,
        price_cfg: dict,
        daily_cfg: dict,
        ln_cfg: dict,
        parity_cfg: dict,
        fonts_cfg: dict,
        legend_cfg: dict,
        fig_dir: str,
        start_year: int,
        toggles: dict
):
    """Generates and saves the historical price and MA ratio plot."""
    os.makedirs(fig_dir, exist_ok=True)

    save_fig = toggles.get("save_fig", True)
    without_legend = toggles.get("without_legend", False)
    only_legend = toggles.get("only_legend", False)
    split = toggles.get("split", False)
    historic_down = toggles.get("historic_down", False)
    also_daily = toggles.get("also_daily", False)
    include_ln = toggles.get("include_ln", False)
    price_lowest = toggles.get("price_lowest", toggles.get("price_bottom", toggles.get("ln_middle", False)))

    axis_label_size = fonts_cfg.get("axis_label_size", 18)
    tick_label_size = fonts_cfg.get("tick_label_size", 14)

    # --- LEGEND PLACEMENTS & FONT SIZES ---
    price_legend_loc = legend_cfg.get("price_loc", legend_cfg.get("price_position", "upper left"))
    ratio_legend_loc = legend_cfg.get("ratio_loc", legend_cfg.get("ratio_position", "upper left"))
    ln_legend_loc = legend_cfg.get("ln_loc", legend_cfg.get("ln_position", "upper left"))

    default_legend_size = legend_cfg.get("font_size", fonts_cfg.get("legend_font_size", 10))
    price_font_size = legend_cfg.get("price_font_size", default_legend_size)
    ratio_font_size = legend_cfg.get("ratio_font_size", default_legend_size)
    ln_font_size = legend_cfg.get("ln_font_size", default_legend_size)

    # --- SETUP FIGURE & AXES ---
    ax_ln = None
    if include_ln:
        if price_lowest:
            # 3 Rows: 1. MA Ratio (top), 2. ln(Price) (middle), 3. Price (bottom)
            fig, (ax_ratio, ax_ln, ax_price) = plt.subplots(
                3, 1, figsize=(12, 11), dpi=150, sharex=True,
                gridspec_kw={"height_ratios": [1.2, 1, 1]}
            )
            bottom_axis = ax_price
        else:
            # 3 Rows: 1. MA Ratio (top), 2. Price (middle), 3. ln(Price) (bottom)
            fig, (ax_ratio, ax_price, ax_ln) = plt.subplots(
                3, 1, figsize=(12, 11), dpi=150, sharex=True,
                gridspec_kw={"height_ratios": [1.2, 1, 1]}
            )
            bottom_axis = ax_ln
    elif split:
        if historic_down:
            fig, (ax_ratio, ax_price) = plt.subplots(
                2, 1, figsize=(12, 8), dpi=150, sharex=True,
                gridspec_kw={"height_ratios": [1.2, 1]}
            )
            bottom_axis = ax_price
        else:
            fig, (ax_price, ax_ratio) = plt.subplots(
                2, 1, figsize=(12, 8), dpi=150, sharex=True,
                gridspec_kw={"height_ratios": [1, 1.2]}
            )
            bottom_axis = ax_ratio
    else:
        fig, ax_price = plt.subplots(figsize=(12, 6), dpi=150)
        ax_ratio = ax_price.twinx()
        bottom_axis = ax_price

    # --- 1. PLOT DAILY PRICE (ax_price) ---
    if also_daily and df_daily is not None and not df_daily.empty:
        ax_price.plot(
            df_daily.index,
            df_daily["price"],
            color=daily_cfg.get("color", "lightcoral"),
            linestyle=daily_cfg.get("linestyle", "-"),
            linewidth=daily_cfg.get("linewidth", 0.8),
            alpha=daily_cfg.get("alpha", 0.45),
            label=daily_cfg.get("label", "EUA Daily Price"),
            zorder=1
        )

    # --- 2. PLOT MONTHLY AVERAGE PRICE (ax_price) ---
    price_color = price_cfg.get("color", "red")
    ax_price.plot(
        df_monthly.index,
        df_monthly["price"],
        color=price_color,
        linestyle=price_cfg.get("linestyle", "-"),
        linewidth=price_cfg.get("linewidth", 2.5),
        label=price_cfg.get("label", "EUA Monthly Avg Price"),
        zorder=3
    )
    ax_price.set_ylabel("EUA Price (€)", color=price_color, fontsize=axis_label_size, fontweight="bold")
    ax_price.tick_params(axis="y", labelcolor=price_color, labelsize=tick_label_size)
    ax_price.tick_params(axis="x", labelsize=tick_label_size)
    ax_price.set_ylim(bottom=0)
    ax_price.grid(True, linestyle="--", alpha=0.4)

    # --- 3. PLOT LN OF PRICES (ax_ln) ---
    if include_ln and ax_ln is not None:
        if also_daily and df_daily is not None and not df_daily.empty:
            ln_daily_label = daily_cfg.get("ln_label", f"ln({daily_cfg.get('label', 'EUA Daily Price')})")
            ax_ln.plot(
                df_daily.index,
                np.log(df_daily["price"]),
                color=daily_cfg.get("color", "lightcoral"),
                linestyle=daily_cfg.get("linestyle", "-"),
                linewidth=daily_cfg.get("linewidth", 0.8),
                alpha=daily_cfg.get("alpha", 0.45),
                label=ln_daily_label,
                zorder=1
            )

        ln_monthly_label = price_cfg.get("ln_label", f"ln({price_cfg.get('label', 'EUA Monthly Avg Price')})")
        ax_ln.plot(
            df_monthly.index,
            np.log(df_monthly["price"]),
            color=price_color,
            linestyle=price_cfg.get("linestyle", "-"),
            linewidth=price_cfg.get("linewidth", 2.5),
            label=ln_monthly_label,
            zorder=3
        )

        ln_ylabel = ln_cfg.get("ylabel", r"$\ln(\mathrm{EUA\ Price})$")
        ax_ln.set_ylabel(ln_ylabel, color=price_color, fontsize=axis_label_size, fontweight="bold")
        ax_ln.tick_params(axis="y", labelcolor=price_color, labelsize=tick_label_size)
        ax_ln.tick_params(axis="x", labelsize=tick_label_size)
        ax_ln.grid(True, linestyle="--", alpha=0.4)

    # --- 4. PLOT MA RATIOS (ax_ratio) ---
    for cfg in ma_configs:
        ax_ratio.plot(
            df_monthly.index,
            df_monthly[f"ratio_{cfg['p1']}_{cfg['p2']}"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            label=cfg["label"]
        )

    ax_ratio.set_ylabel("MA Ratio", color="#333333", fontsize=axis_label_size, fontweight="bold")
    ax_ratio.tick_params(axis="y", labelcolor="#333333", labelsize=tick_label_size)
    ax_ratio.tick_params(axis="x", labelsize=tick_label_size)
    ax_ratio.grid(True, linestyle="--", alpha=0.4)

    # --- 5. PARITY BASELINE AT 1.0 ---
    if parity_cfg.get("show", True):
        y_val = parity_cfg.get("y", 1.0)

        # Line (zorder=5 so it sits on top of the text box)
        ax_ratio.axhline(
            y=y_val,
            color=parity_cfg.get("color", "black"),
            linestyle=parity_cfg.get("linestyle", "-"),
            linewidth=parity_cfg.get("linewidth", 1.6),
            alpha=parity_cfg.get("alpha", 0.7),
            zorder=5
        )

        # Directional labels (zorder=4)
        if parity_cfg.get("show_labels", True):
            yaxis_tf = ax_ratio.get_yaxis_transform()
            x_pos = parity_cfg.get("x_pos", 0.015)
            font_size = parity_cfg.get("font_size", 10)
            font_weight = parity_cfg.get("font_weight", "bold")
            label_color = parity_cfg.get("label_color", "black")
            label_alpha = parity_cfg.get("label_alpha", 0.85)

            bbox_dict = None
            if parity_cfg.get("bbox_show", True):
                bbox_dict = dict(
                    boxstyle=f"round,pad={parity_cfg.get('bbox_pad', 0.2)}",
                    facecolor=parity_cfg.get("bbox_color", "white"),
                    edgecolor="none",
                    alpha=parity_cfg.get("bbox_alpha", 0.50)
                )

            # Upper label (Increase)
            y_up = y_val + parity_cfg.get("y_offset_upper", 0.03)
            text_up = parity_cfg.get("label_upper", "▲ Increase")
            ax_ratio.text(
                x_pos, y_up, text_up,
                transform=yaxis_tf, color=label_color, fontsize=font_size,
                fontweight=font_weight, va="bottom", ha="left", alpha=label_alpha,
                bbox=bbox_dict,
                zorder=4
            )

            # Lower label (Reduction)
            y_down = y_val - parity_cfg.get("y_offset_lower", 0.03)
            text_down = parity_cfg.get("label_lower", "▼ Reduction")
            ax_ratio.text(
                x_pos, y_down, text_down,
                transform=yaxis_tf, color=label_color, fontsize=font_size,
                fontweight=font_weight, va="top", ha="left", alpha=label_alpha,
                bbox=bbox_dict,
                zorder=4
            )

    # --- 6. THRESHOLDS & TRIGGERS (ax_ratio) ---
    for th in thresholds:
        start = pd.to_datetime(th["start_date"]) if "start_date" in th else df_monthly.index.min()
        end = pd.to_datetime(th["end_date"]) if "end_date" in th else df_monthly.index.max()

        mask = (df_monthly.index >= start) & (df_monthly.index <= end)
        segment_dates = df_monthly.index[mask]

        if len(segment_dates) == 0:
            continue

        if th.get("type") == "step":
            th_series = pd.Series(th["y_before"], index=segment_dates)
            switch_date = pd.to_datetime(th["switch_date"])
            th_series[th_series.index >= switch_date] = th["y_after"]

            ax_ratio.step(
                th_series.index, th_series, where="post",
                color=th["color"], linestyle=th["linestyle"],
                linewidth=th["linewidth"], alpha=0.6, label=th["label"]
            )
        else:
            ax_ratio.plot(
                segment_dates, [th["y"]] * len(segment_dates),
                color=th["color"], linestyle=th["linestyle"],
                linewidth=th["linewidth"], alpha=0.6, label=th["label"]
            )

    # --- FORMATTING X-AXIS ---
    bottom_axis.set_xlabel("Date (Months)", fontsize=axis_label_size)
    bottom_axis.xaxis.set_major_locator(mdates.YearLocator())
    bottom_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(bottom_axis.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # --- COLLECT HANDLES / LABELS ---
    handles_price, labels_price = ax_price.get_legend_handles_labels()
    handles_ratio, labels_ratio = ax_ratio.get_legend_handles_labels()
    handles_ln, labels_ln = (ax_ln.get_legend_handles_labels() if ax_ln is not None else ([], []))

    # --- ONLY_LEGEND HANDLER ---
    if only_legend:
        combined_handles = handles_price + handles_ratio + handles_ln
        combined_labels = labels_price + labels_ratio + labels_ln
        legend_save_path = os.path.join(fig_dir, f"historical_legend_only_{start_year}.svg")
        _save_standalone_legend(fig, combined_handles, combined_labels, legend_save_path, save_fig, font_size=default_legend_size)
        return

    # --- ATTACH SEPARATE LEGENDS TO RESPECTIVE AXES ---
    if not without_legend:
        # 1. Price Legend on ax_price
        if handles_price:
            ax_price.legend(
                handles_price,
                labels_price,
                loc=price_legend_loc,
                fontsize=price_font_size,
                framealpha=0.9
            )

        # 2. Ratio Legend on ax_ratio
        if handles_ratio:
            ax_ratio.legend(
                handles_ratio,
                labels_ratio,
                loc=ratio_legend_loc,
                fontsize=ratio_font_size,
                framealpha=0.9
            )

        # 3. ln(Price) Legend on ax_ln (if enabled)
        if include_ln and handles_ln:
            ax_ln.legend(
                handles_ln,
                labels_ln,
                loc=ln_legend_loc,
                fontsize=ln_font_size,
                framealpha=0.9
            )

    plt.tight_layout()

    # --- SAVE FIGURE ---
    if save_fig:
        split_suffix = "_3panel" if include_ln else ("_split" if split else "")
        order_suffix = "_price_lowest" if (include_ln and price_lowest) else ""
        legend_suffix = "_no_legend" if without_legend else ""
        save_path = os.path.join(
            fig_dir,
            f"historical_price_and_ma_ratios_{start_year}{split_suffix}{order_suffix}{legend_suffix}.svg"
        )
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        print(f"[SUCCESS] Saved figure to: {save_path}")

    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    config = load_config()
    start_year = config["data"]["start_year"]
    out_dir = config["paths"]["output_data_dir"]
    fig_dir = config["paths"]["output_figures"]
    toggles = config["plotting"]["toggles"]

    price_cfg = config["plotting"].get("historical_price", {
        "color": "red",
        "linestyle": "-",
        "linewidth": 2.5,
        "label": "EUA Monthly Avg Price"
    })

    daily_cfg = config["plotting"].get("daily_price", {
        "color": "lightcoral",
        "linestyle": "-",
        "linewidth": 0.8,
        "alpha": 0.45,
        "label": "EUA Daily Price"
    })

    ln_cfg = config["plotting"].get("ln_price", {
        "ylabel": r"$\ln(\mathrm{EUA\ Price})$"
    })

    ma_configs = config["plotting"]["ma_ratios"]
    thresholds = config["plotting"]["thresholds"]
    parity_cfg = config["plotting"].get("parity_line", {})
    fonts_cfg = config["plotting"].get("fonts", {
        "axis_label_size": 18,
        "tick_label_size": 14,
        "legend_font_size": 10
    })

    # Load legend configuration (defaulting to empty dict if not present)
    legend_cfg = config["plotting"].get("legend", {})

    input_csv = os.path.join(out_dir, f"merged_eua_{start_year}_2026.csv")

    print(f"Loading data from: {input_csv}")
    df_monthly, df_daily = load_and_calculate_ratios(input_csv, ma_configs)

    print("Generating plot...")
    plot_historical_ratios(
        df_monthly=df_monthly,
        df_daily=df_daily,
        ma_configs=ma_configs,
        thresholds=thresholds,
        price_cfg=price_cfg,
        daily_cfg=daily_cfg,
        ln_cfg=ln_cfg,
        parity_cfg=parity_cfg,
        fonts_cfg=fonts_cfg,
        legend_cfg=legend_cfg,
        fig_dir=fig_dir,
        start_year=start_year,
        toggles=toggles
    )


if __name__ == "__main__":
    main()