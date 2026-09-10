import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


# ==============================================================================
# ANALYTICAL SOLVER & DIAGNOSTICS
# ==============================================================================
def solve_max_growth_r(p1: int, p2: int, m: float) -> float:
    """
    Solves for the monthly growth factor r such that:
        m = (p1 / p2) * (r**(p1 + p2) - r**p1) / (r**p1 - 1),  m > 0
    with m = 1 <=> r = 1.
    """
    if np.isclose(m, 1.0):
        return 1.0

    def objective(r: float) -> float:
        if np.isclose(r, 1.0):
            return 1.0 - m
        val = (float(p1) / float(p2)) * (r ** (p1 + p2) - r ** p1) / (r ** p1 - 1.0)
        return val - m

    if m > 1.0:
        low, high = 1.0 + 1e-12, 2.0
        while objective(high) < 0:
            high *= 2.0
            if high > 1e8:
                break
    else:
        low, high = 1e-12, 1.0 - 1e-12
        while objective(low) > 0:
            low /= 10.0
            if low < 1e-30:
                break

    return float(brentq(objective, low, high, xtol=1e-12, rtol=1e-12))


def log_warmup_diagnostics(p1: int, p2: int, m: float, mode: str, S0: float = 100.0) -> dict:
    """Computes analytical warm-up sequence and verifies MA ratio equality."""
    detect_reduction = (mode == "reduction")
    m_target = (1.0 / m) if detect_reduction else m
    r = solve_max_growth_r(p1, p2, m_target)
    r_annual = r ** 12.0

    t = np.arange(p1 + p2)
    S_det = S0 * (r ** t)

    ma_p1 = np.mean(S_det[0:p1])
    ma_p2 = np.mean(S_det[p1:p1 + p2])
    eval_ratio = ma_p2 / ma_p1

    return {
        "p1": p1,
        "p2": p2,
        "m_param": m,
        "m_target": m_target,
        "r_monthly": r,
        "r_annual_X_crit": r_annual,
        "ma_p1": ma_p1,
        "ma_p2": ma_p2,
        "eval_ratio": eval_ratio,
        "discrepancy": abs(eval_ratio - m_target),
        "warmup_series": S_det[:-1],
        "eval_step_S": S_det[-1]
    }


# ==============================================================================
# DRIFT PARAMETERS & OVERSAMPLING BOUNDS
# ==============================================================================
def get_daily_c_for_target_X(
        X_target: float,
        delta_M: int,
        vol_pct: float,
        phi_daily: float = 0.0,
        days_per_month: int = 21
) -> tuple[float, float]:
    """Calculates daily log-drift needed to achieve annual target growth factor X."""
    delta_D = delta_M * days_per_month
    sigma_daily = (vol_pct / 100.0) / np.sqrt(days_per_month)
    log_drift_needed = (np.log(X_target) / delta_D) - (0.5 * (sigma_daily ** 2))
    c_daily = (1.0 - phi_daily) * log_drift_needed
    return c_daily, sigma_daily


def calculate_validated_oversample_bounds(
        x_min: float,
        x_max: float,
        max_vol_pct: float,
        n_eval_months: int,
        threshold_pct: float = 1.0,
        mode: str = "increase"
) -> tuple[float, float]:
    """Computes target parameter expansion bounds for binned ex-post analysis."""
    sigma_annual = (max_vol_pct / 100.0) * np.sqrt(12.0 / n_eval_months)
    tail_prob = max(0.0001, threshold_pct / 100.0)
    z_stat = norm.ppf(1.0 - tail_prob)
    expansion_factor = float(np.exp(z_stat * sigma_annual))

    if mode == "reduction":
        return round(max(0.005, x_min / expansion_factor), 3), round(min(5.0, x_max * expansion_factor), 3)
    return round(max(0.1, x_min / expansion_factor), 3), round(x_max * expansion_factor, 3)


# ==============================================================================
# EVALUATION OF MA TRIGGERS (MONTHLY & DAILY)
# ==============================================================================
def evaluate_ma_violations(
        S: np.ndarray,
        p1: int,
        p2: int,
        m: float,
        detect_reduction: bool = False
) -> np.ndarray:
    """Monthly evaluator: Evaluates MA ratio month-by-month on S_monthly matrix."""
    N_months, n_sims_batch = S.shape
    violation_counts = np.zeros(n_sims_batch, dtype=np.int32)

    for t in range(p1 + p2, N_months):
        ma_p1 = np.mean(S[t - p2 - p1: t - p2, :], axis=0)
        ma_p2 = np.mean(S[t - p2: t, :], axis=0)
        ratio = ma_p2 / ma_p1

        is_violation = ratio < (1.0 / m) if detect_reduction else ratio > m
        violation_counts += is_violation.astype(np.int32)

    return violation_counts


def evaluate_ma_violations_daily(
        S_daily: np.ndarray,
        p1: int,
        p2: int,
        m: float,
        days_per_month: int = 21,
        detect_reduction: bool = False
) -> np.ndarray:
    """Daily evaluator: Evaluates rolling MA ratios on every single day using O(1) prefix sums."""
    D1 = p1 * days_per_month
    D2 = p2 * days_per_month
    total_lookback = D1 + D2
    N_days = S_daily.shape[0] - 1

    C = np.zeros_like(S_daily)
    np.cumsum(S_daily[1:], axis=0, out=C[1:])

    eval_slice = slice(total_lookback, N_days + 1)
    p2_start = slice(total_lookback - D2, N_days + 1 - D2)
    p1_start = slice(total_lookback - D2 - D1, N_days + 1 - D2 - D1)

    ma_p2 = (C[eval_slice, :] - C[p2_start, :]) / float(D2)
    ma_p1 = (C[p2_start, :] - C[p1_start, :]) / float(D1)

    ratio = ma_p2 / ma_p1
    is_violation = ratio < (1.0 / m) if detect_reduction else ratio > m

    return np.sum(is_violation, axis=0, dtype=np.int32)


# ==============================================================================
# PATH SIMULATORS (AR-GARCH & LOGNORMAL)
# ==============================================================================
def simulate_ar_garch_daily_batch(
        N_months: int,
        batch_size: int,
        S0: float,
        c_daily: float,
        phi_daily: float,
        omega_daily: float,
        alpha: float,
        beta: float,
        days_per_month: int = 21,
        dtype=np.float64,
        return_daily: bool = False,
        warm_up_max: bool = False,
        warm_up_r: float = 1.0,
        n_warm_up_months: int = 0
):
    """Simulates daily AR(1)-GARCH(1,1) price paths."""
    N_days = N_months * days_per_month
    y = np.zeros((N_days + 1, batch_size), dtype=dtype)
    sigma2 = np.zeros((N_days + 1, batch_size), dtype=dtype)
    S_daily = np.zeros((N_days + 1, batch_size), dtype=dtype)

    uncond_daily_var = omega_daily / (1.0 - alpha - beta) if (alpha + beta < 1.0) else 0.026 ** 2
    sigma2[0, :] = uncond_daily_var
    S_daily[0, :] = S0

    if warm_up_max and n_warm_up_months > 0:
        N_warm_days = n_warm_up_months * days_per_month
        r_daily = warm_up_r ** (1.0 / days_per_month)
        day_indices = np.arange(1, N_warm_days + 1, dtype=dtype)[:, None]
        S_daily[1:N_warm_days + 1, :] = S0 * (r_daily ** day_indices)
        y[1:N_warm_days + 1, :] = np.log(r_daily)
        sigma2[1:N_warm_days + 1, :] = uncond_daily_var
        start_sim_day = N_warm_days + 1
    else:
        start_sim_day = 1

    u = np.random.normal(0, 1, size=(N_days + 1, batch_size)).astype(dtype)

    for d in range(start_sim_day, N_days + 1):
        y[d, :] = c_daily + phi_daily * y[d - 1, :] + np.sqrt(sigma2[d - 1, :]) * u[d, :]
        S_daily[d, :] = S_daily[d - 1, :] * np.exp(y[d, :])
        eps = y[d, :] - (c_daily + phi_daily * y[d - 1, :])
        sigma2[d, :] = omega_daily + alpha * (eps ** 2) + beta * sigma2[d - 1, :]

    S_monthly_avg = np.zeros((N_months, batch_size), dtype=dtype)
    for m_idx in range(N_months):
        start_day = m_idx * days_per_month + 1
        end_day = (m_idx + 1) * days_per_month + 1
        S_monthly_avg[m_idx, :] = np.mean(S_daily[start_day:end_day, :], axis=0)

    if return_daily:
        return S_monthly_avg, S_daily
    return S_monthly_avg


def simulate_lognormal_daily_batch(
        N_months: int,
        batch_size: int,
        S0: float,
        mu_daily: float,
        sigma_daily: float,
        days_per_month: int = 21,
        dtype=np.float64,
        return_daily: bool = False,
        warm_up_max: bool = False,
        warm_up_r: float = 1.0,
        n_warm_up_months: int = 0
):
    """Vectorized Constant Volatility Lognormal / Geometric Brownian Motion (GBM)."""
    N_days = N_months * days_per_month
    S_daily = np.zeros((N_days + 1, batch_size), dtype=dtype)
    S_daily[0, :] = S0

    y = np.random.normal(loc=mu_daily, scale=sigma_daily, size=(N_days + 1, batch_size)).astype(dtype)
    y[0, :] = 0.0

    if warm_up_max and n_warm_up_months > 0:
        N_warm_days = n_warm_up_months * days_per_month
        r_daily = warm_up_r ** (1.0 / days_per_month)
        day_indices = np.arange(1, N_warm_days + 1, dtype=dtype)[:, None]
        S_daily[1:N_warm_days + 1, :] = S0 * (r_daily ** day_indices)

        y[1:N_warm_days + 1, :] = 0.0
        y_cumsum = np.cumsum(y[N_warm_days + 1:], axis=0)
        S_daily[N_warm_days + 1:, :] = S_daily[N_warm_days, :] * np.exp(y_cumsum)
    else:
        S_daily[1:, :] = S0 * np.exp(np.cumsum(y[1:], axis=0))

    S_monthly_avg = np.zeros((N_months, batch_size), dtype=dtype)
    for m_idx in range(N_months):
        start_day = m_idx * days_per_month + 1
        end_day = (m_idx + 1) * days_per_month + 1
        S_monthly_avg[m_idx, :] = np.mean(S_daily[start_day:end_day, :], axis=0)

    if return_daily:
        return S_monthly_avg, S_daily
    return S_monthly_avg


# ==============================================================================
# BINNED STATISTICS AGGREGATOR (FOR 02b REALIZED BINNED SCRIPT)
# ==============================================================================
def compute_binned_path_statistics(
        all_X: np.ndarray,
        all_T: np.ndarray,
        all_C: np.ndarray,
        x_min: float,
        x_max: float,
        vol_pct: float,
        omega_daily: float,
        num_bins: int = 100
) -> pd.DataFrame:
    """Aggregates pooled Monte Carlo paths into realized annual growth bins."""
    bin_edges = np.linspace(x_min, x_max, num_bins + 1)
    bin_indices = np.digitize(all_X, bin_edges) - 1
    records = []

    for k in range(num_bins):
        in_bin = bin_indices == k
        count = int(np.sum(in_bin))

        if count >= 10:
            bin_center = float(0.5 * (bin_edges[k] + bin_edges[k + 1]))
            triggered = int(np.sum(all_T[in_bin]))
            violations_sum = int(np.sum(all_C[in_bin]))

            prob_viol = float((triggered / count) * 100.0)
            uncond_avg = float(violations_sum / count)
            cond_avg = float(violations_sum / triggered) if triggered > 0 else np.nan

            records.append({
                "Monthly_Vol_%": vol_pct,
                "Base_Variance_omega_daily": round(omega_daily, 10),
                "Bin_Index": k,
                "Realized_Yearly_X": round(bin_center, 4),
                "Path_Count": count,
                "Violation_Prob_%": round(prob_viol, 2),
                "Avg_Violations_Per_Series": round(uncond_avg, 4),
                "Avg_Violations_Conditional": round(cond_avg, 4),
            })

    return pd.DataFrame(records)