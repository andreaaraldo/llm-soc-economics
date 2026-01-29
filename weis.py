"""
SOC LLM Economy — paper-faithful Monte Carlo simulator (WEIS LLM Economy paper)

What this script does
1) Computes baseline daily cost without LLM: C_noLLM
2) Samples random variables (paper):
     - eta_{k,r}(a)     ~ Beta(mean=eta_bar_{k,r}(a), CV)
     - gamma_{k,r}(a)   ~ Gamma(mean=gamma_bar_{k,r}(a), CV)
     - tau_new_{k,r}(a) ~ Gamma(mean=tau_new_bar_{k,r}(a), CV)
   with the mean link (paper Eq.13):
     eta_bar_{k,r}(a) = 1 - exp( -gamma_bar_{k,r}(a) / xi[a] )
   IMPORTANT: Eq.13 links the MEANS; eta and gamma are sampled independently.
3) For each (theta, CV), computes ROI samples and:
     - p_CV(theta) = P(ROI > 0)
     - VaR_5%(ROI) classes chessboard
4) Optional ROI boxplots and ROI-vs-token-price sensitivity.

All figures are saved in the same folder as this script in PNG.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches


# ============================================================
# GLOBAL CONFIGURATION (edit here)
# ============================================================

RANDOM_SEED = 7
rng = np.random.default_rng(RANDOM_SEED)

eta_max = 0.9 # maximum productivity gain achievable via the LLM

FIG_FORMAT = "png"   # best for plots with text
FIG_DPI = 150        # lighter than 300
FIG_FACE = "white"

alert_scale=1
alert_proportion = {"low": 0.75, "med": 0.2, "high": 0.05}  

VAR_LEVEL = 0.01   # Level of the value at risk


# ============================================================
# CAPEX RATIO theta (paper: amortized daily equivalent)
# ============================================================

theta_grid = np.linspace(0.01, 0.50, 40)
theta_bins = np.array([0.05, 0.10, 0.20, 0.30, 0.40])
CV_levels = [0.05, 0.25, 0.50, 1.00]


# ============================================================
# SETTING MEAN PARAMETERS
# ============================================================


# Base mean FTE-day per (task) for each alert type multiplier and role multiplier
fte_scale_old_tasks = 0.02 # base mean FTE spent on a task
TAU_BASE_BY_TASK = {"triage": 0.15, "analysis": 0.6, "reporting": 0.25}
TAU_MULT_BY_ALERT = {"low": 1.0, "med": 1.8, "high": 4.0}
TAU_MULT_BY_ROLE = {"L1": 1.0, "L2": 0.8, "L3": 0.4}

# New-task mean effort (FTE-day) with multipliers
fte_scale_new_tasks = 0.0002 # base mean FTE spent on a task
TAU_NEW_BASE_BY_TASK = {"review": 0.01, "rework": 0.02, "prompting": 0.08}
TAU_NEW_MULT_BY_ALERT = {"low": 1.0, "med": 2.0, "high": 5.0}
TAU_NEW_MULT_BY_ROLE = {"L1": 0.6, "L2": 1.0, "L3": 1.2}


gamma_scale = 1200 # base mean tokens spent on a task
# Base mean tokens per (task) with alert/role multipliers
GAMMA_BASE_BY_TASK = {"triage": 0.14, "analysis": 0.55, "reporting": 0.31} #Fraction of tokens spent
GAMMA_MULT_BY_ALERT = {"low": 1.0, "med": 1.6, "high": 2.4}
GAMMA_MULT_BY_ROLE = {"L1": 0.8, "L2": 1.0, "L3": 1.3}



# ============================================================
# EQ.13 PARAMETER xi[a] (tokens per alert)
# ============================================================
xi_tokens_scale = gamma_scale
xi_tokens: dict[str, float] = {"low": 1.5*xi_tokens_scale, \
"med": 1.0*xi_tokens_scale, "high": 0.5*xi_tokens_scale}



# ============================================================
# MONTE CARLO SIMULATION SETTINGS
# ============================================================

N_SAMPLES_MAIN = 500
N_SAMPLES_BOXPLOT = N_SAMPLES_MAIN

USE_LOG_Y_FOR_PROB = False
PROB_LOG_EPS_PERCENT = 1e-6




# ============================================================
# FROM THIS POINT ONWARD YOU DON'T NEED TO CARE
# ============================================================

def get_script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


FIG_DIR = get_script_dir()


def save_figure(fig_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"{fig_name}.{FIG_FORMAT}", dpi=FIG_DPI, bbox_inches="tight", facecolor=FIG_FACE)


# ============================================================
# SOC WORKLOAD & COSTS (paper symbols)
# ============================================================

ALERT_TYPES = ["low", "med", "high"]          # a ∈ A
ROLES = ["L1", "L2", "L3"]                    # r ∈ R
TASKS_NO_LLM = ["triage", "analysis", "reporting"]  # k ∈ K
TASKS_NEW = ["review", "rework", "prompting"]       # k ∈ K_new


HOURS_PER_DAY = 8.0
hourly_cost_eur = {"L1": 45.0, "L2": 70.0, "L3": 110.0}
cost_per_fte_day_eur = {r: hourly_cost_eur[r] * HOURS_PER_DAY for r in ROLES}  # c_r (€/FTE-day)

C_inv_eur_per_day = 0 ### Set it to zero if you consider C_op\nollm in the formula of theta


# ============================================================
# TOKEN PRICING
# ============================================================

USE_IN_OUT_PRICING = True

EURUSD = 1.1836
USD_to_EUR = 1.0 / EURUSD

USD_PER_1M_INPUT = 2.50
USD_PER_1M_OUTPUT = 10.00

c_tok_in_eur_per_token = (USD_PER_1M_INPUT * USD_to_EUR) / 1e6
c_tok_out_eur_per_token = (USD_PER_1M_OUTPUT * USD_to_EUR) / 1e6

OUTPUT_TOKEN_SHARE = 0.35  # share of output tokens within total tokens

c_tok_eur_per_token = 2.0e-6  # used only if USE_IN_OUT_PRICING=False


def token_cost_eur(total_tokens: np.ndarray) -> np.ndarray:
    if not USE_IN_OUT_PRICING:
        return c_tok_eur_per_token * total_tokens
    out_tokens = total_tokens * OUTPUT_TOKEN_SHARE
    in_tokens = total_tokens - out_tokens
    return c_tok_in_eur_per_token * in_tokens + c_tok_out_eur_per_token * out_tokens


# ============================================================
# UNITS: tau semantics
# ============================================================

TAU_IS_FTE_PER_ALERT = False
# Paper typically treats tau_{k,r}(a) as already aggregated per day (so this should be False).
# If you calibrate tau as "per alert", set True and tau will be multiplied by n_a.

def volume_multiplier(a: str) -> float:
    return float(alert_proportion[a]) if TAU_IS_FTE_PER_ALERT else 1.0

# ============================================================
# SETTING MEAN PARAMETERS
# ============================================================


tau_bar_no_llm: dict[tuple[str, str, str], float] = {}
for a in ALERT_TYPES:
    for k in TASKS_NO_LLM:
        for r in ROLES:
            tau_bar_no_llm[(k, r, a)] = (
                fte_scale_old_tasks
                *TAU_BASE_BY_TASK[k] * TAU_MULT_BY_ALERT[a] * TAU_MULT_BY_ROLE[r]
            )

tau_new_bar: dict[tuple[str, str, str], float] = {}
for a in ALERT_TYPES:
    for k in TASKS_NEW:
        for r in ROLES:
            tau_new_bar[(k, r, a)] = (
                fte_scale_new_tasks
                * TAU_NEW_BASE_BY_TASK[k]
                * TAU_NEW_MULT_BY_ALERT[a]
                * TAU_NEW_MULT_BY_ROLE[r]
            )

gamma_bar: dict[tuple[str, str, str], float] = {}
for a in ALERT_TYPES:
    for k in TASKS_NO_LLM:
        for r in ROLES:
            gamma_bar[(k, r, a)] = (
                gamma_scale*
                GAMMA_BASE_BY_TASK[k] * GAMMA_MULT_BY_ALERT[a] * GAMMA_MULT_BY_ROLE[r]
            )

# ============================================================
# DISTRIBUTIONS (mean + CV)
# ============================================================

def beta_params_from_mean_cv(mean: float, cv: float) -> tuple[float, float]:
    mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    max_cv2 = (1.0 - mean) / mean - 1e-12
    cv2 = min(float(cv) * float(cv), max_cv2)
    s = (1.0 - mean) / (mean * cv2) - 1.0  # α+β
    alpha = max(mean * s, 1e-3)
    beta = max((1.0 - mean) * s, 1e-3)
    return alpha, beta


def gamma_params_from_mean_cv(mean: float, cv: float) -> tuple[float, float]:
    mean = float(max(mean, 1e-12))
    k = 1.0 / (float(cv) * float(cv))
    scale = mean * (float(cv) * float(cv))
    return k, scale


def eta_bar_from_eq13(gamma_mean: float, xi_a: float) -> float:
    if xi_a <= 0:
        raise ValueError(f"xi[a] must be > 0. Got {xi_a}.")
    eta = eta_max * (1.0 - math.exp(-float(gamma_mean) / float(xi_a)) )
    return float(np.clip(eta, 1e-6, 1.0 - 1e-6))


# ============================================================
# COST MODEL
# ============================================================

def compute_C_no_llm() -> float:
    personnel = 0.0
    for a in ALERT_TYPES:
        mult = volume_multiplier(a)
        for k in TASKS_NO_LLM:
            for r in ROLES:
                tau = tau_bar_no_llm[(k, r, a)]
                personnel += cost_per_fte_day_eur[r] * tau * mult
    return float(personnel + C_inv_eur_per_day)


C_noLLM = compute_C_no_llm()


def sample_ROI_components(theta: float, cv: float, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if theta <= 0:
        raise ValueError("theta must be > 0.")
    if cv <= 0:
        raise ValueError("cv must be > 0.")
    if N <= 0:
        raise ValueError("N must be > 0.")

    # Sample gamma and eta for existing tasks
    gamma_samples: dict[tuple[str, str, str], np.ndarray] = {}
    eta_samples: dict[tuple[str, str, str], np.ndarray] = {}

    for (k, r, a), g_mean in gamma_bar.items():
        shape_g, scale_g = gamma_params_from_mean_cv(g_mean, cv)
        gamma_samples[(k, r, a)] = rng.gamma(shape=shape_g, scale=scale_g, size=N)

        eta_mean = eta_bar_from_eq13(gamma_mean=g_mean, xi_a=xi_tokens[a])
        alpha, beta = beta_params_from_mean_cv(eta_mean, cv)
        eta_samples[(k, r, a)] = rng.beta(alpha, beta, size=N)

    # Sample tau_new for new tasks
    tau_new_samples: dict[tuple[str, str, str], np.ndarray] = {}
    for key, t_mean in tau_new_bar.items():
        shape_t, scale_t = gamma_params_from_mean_cv(t_mean, cv)
        tau_new_samples[key] = rng.gamma(shape=shape_t, scale=scale_t, size=N)

    baseline_personnel = C_noLLM - C_inv_eur_per_day

    llm_personnel_existing = np.zeros(N, dtype=float)
    llm_personnel_new = np.zeros(N, dtype=float)
    token_cost_total = np.zeros(N, dtype=float)

    for a in ALERT_TYPES:
        mult = volume_multiplier(a)
        n_a = float(alert_proportion[a])

        for k in TASKS_NO_LLM:
            for r in ROLES:
                key = (k, r, a)
                tau_component = tau_bar_no_llm[key] * mult
                llm_personnel_existing += cost_per_fte_day_eur[r] * tau_component * (1.0 - eta_samples[key])
                token_cost_total += token_cost_eur(gamma_samples[key] * n_a)

        for k in TASKS_NEW:
            for r in ROLES:
                key = (k, r, a)
                tau_new_component = tau_new_samples[key] * mult
                llm_personnel_new += cost_per_fte_day_eur[r] * tau_new_component

    Delta_eff = baseline_personnel - llm_personnel_existing
    Beta_tok = token_cost_total
    Beta_new = llm_personnel_new
    DeltaC = Delta_eff - Beta_tok - Beta_new

    C_cap = theta * C_noLLM
    ROI = (DeltaC - C_cap) / max(C_cap, 1e-12)

    return ROI, DeltaC, Delta_eff, Beta_tok, Beta_new


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_sanity(theta_ref: float = 0.10, cv_ref: float = 0.25, N: int = 20_000) -> None:
    ROI, DeltaC, Delta_eff, Beta_tok, Beta_new = sample_ROI_components(theta_ref, cv_ref, N)
    C_cap = theta_ref * C_noLLM

    print("SANITY CHECK (medians):")
    print(f"  C_noLLM (det)      = {C_noLLM:,.2f} €/day")
    print(f"  theta (paper)      = {theta_ref:.3f}  -> C_cap = {C_cap:,.2f} €/day")
    print(f"  median Delta_eff   = {np.median(Delta_eff):+.2f} €/day")
    print(f"  median Beta_tok    = {np.median(Beta_tok):+.2f} €/day")
    print(f"  median Beta_new    = {np.median(Beta_new):+.2f} €/day")
    print(f"  median DeltaC      = {np.median(DeltaC):+.2f} €/day")
    print(f"  median ROI         = {np.median(ROI):+.4f}")
    print(f"  P(ROI>0)           = {np.mean(ROI > 0.0)*100:.3f}%")
    print("")


# ============================================================
# FIG.1: p_CV(theta)
# ============================================================

def plot_profitability_probability() -> None:
    p_profit: dict[float, list[float]] = {cv: [] for cv in CV_levels}

    for cv in CV_levels:
        for theta in theta_grid:
            ROI, *_ = sample_ROI_components(theta=theta, cv=cv, N=N_SAMPLES_MAIN)
            p_profit[cv].append(float(np.mean(ROI > 0.0)))

    plt.figure()
    for cv in CV_levels:
        y_percent = np.array(p_profit[cv]) * 100.0
        if USE_LOG_Y_FOR_PROB:
            y_percent = np.maximum(y_percent, PROB_LOG_EPS_PERCENT)
        plt.plot(theta_grid * 100.0, y_percent, label=f"CV={cv}")

    plt.xlabel(r"$\theta$ ")
    plt.ylabel(r"$p_{CV}(\theta)=P(ROI>0)$ ")
    if USE_LOG_Y_FOR_PROB:
        plt.yscale("log")
        plt.ylim(PROB_LOG_EPS_PERCENT, 100.0)
        plt.title("Profitability probability vs CAPEX ratio (log-y)")
    else:
        plt.ylim(0.0, 100.0)
        plt.title("Profitability probability vs CAPEX ratio")

    plt.legend()
    plt.tight_layout()
    save_figure("fig1_p_profit_vs_theta")
    plt.show()


# ============================================================
# FIG.2: Chessboard VaR_5%(ROI) classes
# ============================================================

def plot_chessboard_var_roi(var_level: float = 0.05) -> None:
    if not (0.0 < var_level < 1.0):
        raise ValueError("var_level must be in (0,1).")

    V = np.zeros((len(CV_levels), len(theta_bins)), dtype=float)

    for i, cv in enumerate(CV_levels):
        for j, theta in enumerate(theta_bins):
            ROI, *_ = sample_ROI_components(theta=theta, cv=cv, N=N_SAMPLES_MAIN)
            # VaR_z(ROI) = -quantile_z(ROI), expressed in %
            V[i, j] = -np.quantile(ROI, var_level) * 100.0

    bounds = [0, 5, 10, 30, 50, 1e9]
    cmap = ListedColormap(["white", "0.85", "0.70", "0.45", "0.15"])
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure()
    plt.imshow(V, aspect="auto", norm=norm, cmap=cmap, origin="lower")
    plt.xticks(np.arange(len(theta_bins)), [f"{int(t*100)}%" for t in theta_bins])
    plt.yticks(np.arange(len(CV_levels)), [f"CV={cv}" for cv in CV_levels])
    plt.xlabel(r"$\theta$")
    plt.ylabel("Uncertainty level (CV)")

    z_pct = int(round(var_level * 100))
    plt.title(rf"Chessboard: $VaR_{{{z_pct}\%}}(ROI)$ classes (in %)")

    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, len(theta_bins), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(CV_levels), 1), minor=True)
    plt.grid(which="minor", linestyle="-", linewidth=1)
    plt.tick_params(which="minor", bottom=False, left=False)

    labels = ["0–5%", "5–10%", "10–30%", "30–50%", ">50%"]
    patches = [mpatches.Patch(color=cmap(i), label=labels[i]) for i in range(cmap.N)]
    plt.legend(
        handles=patches,
        title=rf"$VaR_{{{z_pct}\%}}(ROI)$",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    plt.tight_layout()
    save_figure(f"fig_chessboard_var_roi_{z_pct}pct")
    plt.show()




# ============================================================
# FIG.NEW: Boxplots of DeltaC components per (theta, CV)
#   DeltaC = Delta_eff - Beta_tok - Beta_new
#   We plot: Delta_eff, -Beta_tok, -Beta_new  (signed contributions)
# ============================================================

def plot_boxplots_deltaC_components_per_config() -> None:
    for cv in CV_levels:
        for theta in theta_bins:
            _, _, Delta_eff, Beta_tok, Beta_new = sample_ROI_components(
                theta=theta, cv=cv, N=N_SAMPLES_BOXPLOT
            )

            data = [Delta_eff, -Beta_tok, -Beta_new]
            labels = [r"$\Delta^{eff}$", r"$-\beta^{tok}$", r"$-\beta^{new}$"]

            plt.figure()
            plt.boxplot(data, labels=labels, showfliers=False, whis=(5, 95))
            plt.axhline(0.0, linestyle="--", linewidth=1)
            plt.ylabel("€/day (signed contribution to ΔC)")
            plt.title(
                f"ΔC components (signed) — theta={int(theta*100)}%, CV={cv}, N={N_SAMPLES_BOXPLOT}"
            )
            plt.tight_layout()
            save_figure(
                f"box_deltaC_components_theta_{int(theta*100)}_cv_{str(cv).replace('.', 'p')}"
            )
            plt.show()



# ============================================================
# EXTRA: ROI boxplots
# ============================================================

def plot_boxplots_roi_across_theta() -> None:
    for cv in CV_levels:
        data = []
        for theta in theta_bins:
            ROI, *_ = sample_ROI_components(theta=theta, cv=cv, N=N_SAMPLES_BOXPLOT)
            data.append(ROI)

        plt.figure()
        plt.boxplot(data, showfliers=False, whis=(5, 95))
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xticks(np.arange(1, len(theta_bins) + 1), [f"{int(t*100)}%" for t in theta_bins])
        plt.xlabel(r"$\theta$")
        plt.ylabel("ROI (daily)")
        plt.title(f"ROI distribution across CAPEX ratio (CV={cv}, N={N_SAMPLES_BOXPLOT})")
        plt.tight_layout()
        save_figure(f"box_roi_vs_theta_cv_{str(cv).replace('.', 'p')}")
        plt.show()


def plot_boxplots_roi_across_cv() -> None:
    for theta in theta_bins:
        data = []
        for cv in CV_levels:
            ROI, *_ = sample_ROI_components(theta=theta, cv=cv, N=N_SAMPLES_BOXPLOT)
            data.append(ROI)

        plt.figure()
        plt.boxplot(data, showfliers=False, whis=(5, 95))
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xticks(np.arange(1, len(CV_levels) + 1), [f"{cv}" for cv in CV_levels])
        plt.xlabel("CV")
        plt.ylabel("ROI (daily)")
        plt.title(f"ROI distribution across uncertainty levels (theta={int(theta*100)}%, N={N_SAMPLES_BOXPLOT})")
        plt.tight_layout()
        save_figure(f"box_roi_vs_cv_theta_{int(theta*100)}")
        plt.show()


# ============================================================
# EXTRA: ROI vs token price (scales in/out prices together)
# ============================================================

def plot_roi_vs_token_price(theta_ref: float = 0.10, cv_ref: float = 0.25) -> None:
    token_mult_grid = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])

    global c_tok_in_eur_per_token, c_tok_out_eur_per_token, c_tok_eur_per_token
    c_in_0 = c_tok_in_eur_per_token
    c_out_0 = c_tok_out_eur_per_token
    c_single_0 = c_tok_eur_per_token

    med, p10, p90, p_prof = [], [], [], []

    for m in token_mult_grid:
        if USE_IN_OUT_PRICING:
            c_tok_in_eur_per_token = c_in_0 * m
            c_tok_out_eur_per_token = c_out_0 * m
        else:
            c_tok_eur_per_token = c_single_0 * m

        ROI, *_ = sample_ROI_components(theta=theta_ref, cv=cv_ref, N=N_SAMPLES_BOXPLOT)
        med.append(np.median(ROI))
        p10.append(np.quantile(ROI, 0.10))
        p90.append(np.quantile(ROI, 0.90))
        p_prof.append(np.mean(ROI > 0.0))

    # restore
    if USE_IN_OUT_PRICING:
        c_tok_in_eur_per_token = c_in_0
        c_tok_out_eur_per_token = c_out_0
    else:
        c_tok_eur_per_token = c_single_0

    if USE_IN_OUT_PRICING:
        x = (c_in_0 * token_mult_grid) * 1e6
        x_label = "Token price (€/1M input tokens) [scaled]"
    else:
        x = (c_single_0 * token_mult_grid) * 1e6
        x_label = "Token price (€/1M tokens) [scaled]"

    plt.figure()
    plt.plot(x, med, marker="o")
    plt.fill_between(x, p10, p90, alpha=0.2)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel("ROI (daily)")
    plt.title(f"ROI vs token price (theta={int(theta_ref*100)}%, CV={cv_ref}, N={N_SAMPLES_BOXPLOT})")
    plt.tight_layout()
    save_figure(f"roi_vs_token_price_theta_{int(theta_ref*100)}_cv_{str(cv_ref).replace('.', 'p')}")
    plt.show()

    plt.figure()
    plt.plot(x, np.array(p_prof) * 100.0, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("P(ROI>0) ")
    plt.title(f"Profitability vs token price (theta={int(theta_ref*100)}%, CV={cv_ref}, N={N_SAMPLES_BOXPLOT})")
    plt.tight_layout()
    save_figure(f"p_prof_vs_token_price_theta_{int(theta_ref*100)}_cv_{str(cv_ref).replace('.', 'p')}")
    plt.show()



# ============================================================
# FIG.NEW: Boxplot of DeltaC per (theta, CV)
# ============================================================

def plot_boxplot_deltaC() -> None:
    for cv in CV_levels:
        for theta in theta_bins:
            _, DeltaC, *_ = sample_ROI_components(theta=theta, cv=cv, N=N_SAMPLES_BOXPLOT)

            plt.figure()
            plt.boxplot([DeltaC], labels=[r"$\Delta C$"], showfliers=False, whis=(5, 95))
            plt.axhline(0.0, linestyle="--", linewidth=1)
            plt.ylabel("€/day")
            plt.title(f"Boxplot of ΔC — theta={int(theta*100)}%, CV={cv}, N={N_SAMPLES_BOXPLOT}")
            plt.tight_layout()
            save_figure(f"box_deltaC_theta_{int(theta*100)}_cv_{str(cv).replace('.', 'p')}")
            plt.show()


# ============================================================
# FIG.NEW: Boxplot of DeltaC vs C_cap vs C_noLLM (same chart)
# ============================================================

def plot_boxplot_DeltaC_vs_Ccap_vs_CnoLLM() -> None:
    for cv in CV_levels:
        for theta in theta_bins:
            _, DeltaC, *_ = sample_ROI_components(theta=theta, cv=cv, N=N_SAMPLES_BOXPLOT)

            C_cap = theta * C_noLLM

            # Make deterministic quantities "boxplot-compatible"
            C_cap_arr = np.full_like(DeltaC, fill_value=C_cap, dtype=float)
            C_noLLM_arr = np.full_like(DeltaC, fill_value=C_noLLM, dtype=float)

            data = [DeltaC, C_cap_arr, C_noLLM_arr]
            labels = [r"$\Delta C$", r"$C_{cap}$", r"$C_{noLLM}$"]

            plt.figure()
            plt.boxplot(data, labels=labels, showfliers=False, whis=(5, 95))
            plt.axhline(0.0, linestyle="--", linewidth=1)
            plt.ylabel("€/day")
            plt.title(f"ΔC vs CAPEX vs baseline — θ={int(theta*100)}%, CV={cv}, N={N_SAMPLES_BOXPLOT}")
            plt.tight_layout()

            save_figure(
                f"box_DeltaC_Ccap_CnoLLM_theta_{int(theta*100)}_cv_{str(cv).replace('.', 'p')}"
            )
            plt.show()


# ============================================================
# FIG.NEW: eta profile vs tokens (Eq.13 mean link)
#   eta_bar(gamma) = eta_max * (1 - exp(-gamma/xi[a]))
# ============================================================

def plot_eta_profile_vs_tokens() -> None:
    # Pick a token range that covers your configured gamma means (and beyond)
    gamma_means = np.array(list(gamma_bar.values()), dtype=float)
    g_min = max(0.0, float(np.min(gamma_means)) * 0.0)   # start at 0
    g_max = float(np.max(gamma_means)) * 3.0            # extend beyond observed means
    g = np.linspace(g_min, max(g_max, 1.0), 400)

    plt.figure()

    # One curve per alert type (because xi depends on a)
    for a in ALERT_TYPES:
        y = np.array([eta_bar_from_eq13(gamma_mean=gi, xi_a=xi_tokens[a]) for gi in g], dtype=float)
        plt.plot(g, y, label=fr"{a} (xi={xi_tokens[a]:.0f})")

    # Also show where your configured gamma means lie (optional but useful)
    plt.scatter(gamma_means, [0.0] * len(gamma_means), marker="|", linewidths=2)

    plt.xlabel(r"Mean tokens $\bar{\gamma}$")
    plt.ylabel(r"Mean efficiency $\bar{\eta}$")
    plt.title(r"Eq.13 mean link: $\bar{\eta}=\eta_{\max}(1-e^{-\bar{\gamma}/\xi[a]})$")
    plt.ylim(0.0, min(1.0, eta_max * 1.05))
    plt.legend()
    plt.tight_layout()
    save_figure("fig_eta_profile_vs_tokens")
    plt.show()


# ============================================================
# FIG.NEW: Boxplots of tokens per (task, alert type, role)
#   For each a: boxplot of gamma_{k,r}(a) (tokens per query / per unit as modeled)
# ============================================================

def plot_boxplots_tokens_per_task_alert_role(N: int | None = None) -> None:
    if N is None:
        N = N_SAMPLES_BOXPLOT

    for cv in CV_levels:
        # --- sample gamma for all triples once per CV ---
        gamma_samples: dict[tuple[str, str, str], np.ndarray] = {}
        for (k, r, a), g_mean in gamma_bar.items():
            shape_g, scale_g = gamma_params_from_mean_cv(g_mean, cv)
            gamma_samples[(k, r, a)] = rng.gamma(shape=shape_g, scale=scale_g, size=N)

        # --- one plot per alert type (a) for readability ---
        for a in ALERT_TYPES:
            data = []
            labels = []

            # consistent ordering: task then role
            for k in TASKS_NO_LLM:
                for r in ROLES:
                    key = (k, r, a)
                    data.append(gamma_samples[key])
                    labels.append(f"{k}-{r}")

            plt.figure(figsize=(max(10, 0.6 * len(labels)), 5))
            plt.boxplot(data, labels=labels, showfliers=False, whis=(5, 95))
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Tokens (per sampled γ)")
            plt.title(f"Tokens distribution per (task, role) — alert={a}, CV={cv}, N={N}")
            plt.tight_layout()

            save_figure(f"box_tokens_task_role_alert_{a}_cv_{str(cv).replace('.', 'p')}")
            plt.show()



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print_sanity(theta_ref=0.10, cv_ref=0.25, N=20_000)
    plot_eta_profile_vs_tokens()
    plot_boxplots_tokens_per_task_alert_role()
    plot_chessboard_var_roi(VAR_LEVEL) 
    plot_boxplot_DeltaC_vs_Ccap_vs_CnoLLM()
    plot_profitability_probability()
    plot_boxplots_deltaC_components_per_config()
    plot_boxplots_roi_across_theta()
    plot_boxplots_roi_across_cv()
    plot_roi_vs_token_price(theta_ref=0.10, cv_ref=0.25)
    print(f"Saved figures to: {FIG_DIR}")




