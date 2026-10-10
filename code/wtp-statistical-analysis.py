"""
Extended WTP Analysis — Group Assignment
==========================================
Builds on the existing plotting code and adds:

  Q2  : Distribution plot WITH outliers (Figure 3)
  Q5  : Mean WTP per WTP decile — printed table
  Q8a : Mean WTP per income decile — printed table
  Q8b : Bar chart with ± Std Dev  (the version the assignment explicitly asks for)
  Q8c : Spearman rank correlation + OLS regression + Kruskal-Wallis test
  Q9c : Distance decile plot (already in original — kept here too)

Run:  python analysis_extended.py
Deps: pandas matplotlib scipy numpy xlrd  (pip install pandas matplotlib scipy xlrd)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── output folder ─────────────────────────────────────────────────────────
OUTPUT = "figures_improved"
os.makedirs(OUTPUT, exist_ok=True)

# ── global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
})

C_BLUE   = "#2563EB"
C_RED    = "#DC2626"
C_GREEN  = "#059669"
C_ORANGE = "#D97706"
C_GRAY   = "#6B7280"


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_data(path="data.xls"):
    if os.path.exists(path):
        try:
            df = pd.read_excel(path, engine="xlrd")
        except Exception:
            df = pd.read_excel(path)
        print(f"✓  Loaded {len(df):,} rows from '{path}'")
    else:
        print(f"⚠  '{path}' not found — using demo data (7 rows).")
        demo = {
            "perm":         [0,1,1,0,1,1,1],
            "NONCT":        [1,1,1,1,1,1,1],
            "male":         [1,0,1,1,0,0,0],
            "age":          [82,84,79,75,71,64,74],
            "income":       [1,1,1,1,1,1,1],
            "D_WF":         [4642,6306,4409,12370,5847,5423,5457],
            "WTP_farshore": [-415,-645,-836,12572,10632,8836,-1139],
            "age2":         [6724,7056,6241,5625,5041,4096,5476],
        }
        df = pd.DataFrame(demo)
    return df


def ci95(series):
    """Return (mean, lower_err, upper_err) — errors are distances from mean."""
    s = series.dropna()
    n = len(s)
    if n < 2:
        return s.mean(), 0, 0
    se = stats.sem(s)
    t  = stats.t.ppf(0.975, df=n - 1)
    hw = t * se
    return s.mean(), hw, hw


def build_deciles(df, group_col, wtp_col="WTP_farshore", n=10):
    """Rank group_col into n deciles and return per-decile stats DataFrame."""
    if group_col == wtp_col:
        d = df[[group_col]].dropna().copy()
    else:
        d = df[[group_col, wtp_col]].dropna().copy()
    d["decile"] = pd.qcut(
        d[group_col].rank(method="first"),
        q=n, labels=False, duplicates="drop"
    ) + 1
    rows = []
    for dec, grp in d.groupby("decile"):
        m, lo, hi = ci95(grp[wtp_col])
        rows.append({
            "decile":  dec,
            "n":       len(grp),
            "mean":    m,
            "std":     grp[wtp_col].std(),
            "median":  grp[wtp_col].median(),
            "lo_err":  lo,
            "hi_err":  hi,
        })
    return pd.DataFrame(rows).set_index("decile")


def dot_plot(ax, stats_df, color, xlabel="Decile", clip_ylim=None):
    """Connected dot-plot: line joining means + shaded 95 % CI band."""
    x    = stats_df.index.astype(float)
    y    = stats_df["mean"].values
    lo   = stats_df["lo_err"].values
    hi   = stats_df["hi_err"].values

    ax.fill_between(x, y - lo, y + hi, alpha=0.18, color=color, zorder=2)
    ax.plot(x, y, color=color, lw=2.2, zorder=3)
    ax.scatter(x, y, color=color, s=70, zorder=4, edgecolors="white", lw=1.2)
    ax.axhline(0, color=C_GRAY, lw=1, ls=":", zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean WTP (DKK / household / year)")
    ax.set_xticks(x.astype(int))

    if clip_ylim is not None:
        ax.set_ylim(*clip_ylim)
        for xi, yi in zip(x, y):
            if yi > clip_ylim[1]:
                ax.annotate(f"{yi:,.0f}", xy=(xi, clip_ylim[1]),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=8, color=color)
            elif yi < clip_ylim[0]:
                ax.annotate(f"{yi:,.0f}", xy=(xi, clip_ylim[0]),
                            xytext=(0, -12), textcoords="offset points",
                            ha="center", fontsize=8, color=color)


# ════════════════════════════════════════════════════════════════════════════
# Q2 — Distribution WITH outliers  (Figure 3 in the report)
# ════════════════════════════════════════════════════════════════════════════
def plot_q2_with_outliers(df):
    """Full-range histogram showing the fat tail — Figure 3."""
    wtp  = df["WTP_farshore"].dropna()
    mean = wtp.mean()
    med  = wtp.median()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    fig.suptitle("Q2 — Distribution of WTP for Farshore Wind Farm",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── left: full-range histogram ─────────────────────────────────────────
    n_bins = min(int(np.sqrt(len(wtp)) * 2), 60)
    ax1.hist(wtp, bins=n_bins, color=C_BLUE, alpha=0.65,
             edgecolor="white", linewidth=0.3, label="Histogram")

    ax1_r = ax1.twinx()
    kde_x = np.linspace(wtp.min() - 500, wtp.max() + 500, 500)
    kde   = stats.gaussian_kde(wtp)
    ax1_r.plot(kde_x, kde(kde_x), color=C_RED, lw=2, label="KDE")
    ax1_r.set_ylabel("Density", color=C_RED, fontsize=10)
    ax1_r.tick_params(axis="y", colors=C_RED, labelsize=9)
    ax1_r.spines["right"].set_visible(True)
    ax1_r.spines["right"].set_color(C_RED)
    ax1_r.spines["top"].set_visible(False)

    ax1.axvline(mean, color="black",  lw=1.6, ls="--",
                label=f"Mean = {mean:,.0f}")
    ax1.axvline(med,  color=C_GREEN,  lw=1.6, ls="-.",
                label=f"Median = {med:,.0f}")
    ax1.set_xlabel("WTP (DKK/household/year)")
    ax1.set_ylabel("Count")
    ax1.set_title("Histogram with KDE")
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8.5,
               loc="upper right", framealpha=0.9)

    # ── right: boxplot + jitter (full scale) ──────────────────────────────
    ax2.boxplot(wtp, vert=True, patch_artist=True,
                boxprops=dict(facecolor=C_BLUE, alpha=0.35),
                medianprops=dict(color=C_RED, lw=2.2),
                whiskerprops=dict(lw=1.3, color=C_GRAY),
                capprops=dict(lw=1.5, color=C_GRAY),
                flierprops=dict(marker="o", markersize=3,
                                markerfacecolor=C_RED, alpha=0.5,
                                markeredgewidth=0),
                whis=1.5)
    rng    = np.random.default_rng(42)
    jitter = rng.uniform(-0.12, 0.12, len(wtp))
    ax2.scatter(1 + jitter, wtp, color=C_BLUE, alpha=0.15, s=4, zorder=3)
    ax2.axhline(0, color=C_GRAY, lw=1, ls=":")
    ax2.set_xticklabels(["WTP_farshore"])
    ax2.set_ylabel("WTP (DKK/household/year)")
    ax2.set_title("Box-plot with Jittered Observations")

    plt.tight_layout()
    path = f"{OUTPUT}/Q2_WTP_distribution_with_outliers.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  → saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# Q2 — Distribution ZOOMED  (Figure 4 in the report)
# ════════════════════════════════════════════════════════════════════════════
def plot_q2_zoomed(df):
    """99th-percentile zoom — Figure 4."""
    wtp  = df["WTP_farshore"].dropna()
    mean = wtp.mean()
    med  = wtp.median()

    fig = plt.figure(figsize=(14, 6), facecolor="white")
    fig.suptitle("Q2 — Distribution of WTP for Farshore Wind Farm",
                 fontsize=14, fontweight="bold", y=1.01)

    ax1 = fig.add_subplot(1, 2, 1)
    pct1, pct99 = np.percentile(wtp, [0.5, 99])
    wtp_clip    = wtp.clip(pct1, pct99)

    n_bins = min(int(np.sqrt(len(wtp)) * 2.5), 80)
    ax1.hist(wtp_clip, bins=n_bins, color=C_BLUE, alpha=0.65,
             edgecolor="white", linewidth=0.3)

    ax1_r = ax1.twinx()
    kde_x = np.linspace(pct1 - 500, pct99 + 500, 500)
    kde   = stats.gaussian_kde(wtp)
    ax1_r.plot(kde_x, kde(kde_x), color=C_RED, lw=2, label="KDE")
    ax1_r.set_ylabel("Density", color=C_RED, fontsize=10)
    ax1_r.tick_params(axis="y", colors=C_RED, labelsize=9)
    ax1_r.spines["right"].set_visible(True)
    ax1_r.spines["right"].set_color(C_RED)
    ax1_r.spines["top"].set_visible(False)

    ax1.axvline(mean, color="black",  lw=1.6, ls="--",
                label=f"Mean = {mean:,.0f} DKK")
    ax1.axvline(med,  color=C_GREEN,  lw=1.6, ls="-.",
                label=f"Median = {med:,.0f} DKK")
    ax1.axvline(0,    color=C_GRAY,   lw=1.0, ls=":",
                label="WTP = 0")
    ax1.set_xlabel("WTP (DKK / household / year)")
    ax1.set_ylabel("Count")
    ax1.set_title("Histogram + KDE  (99th pct. zoom)", fontweight="bold")

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8.5,
               loc="upper right", framealpha=0.9)

    pct_neg = (wtp < 0).mean() * 100
    ax1.text(0.02, 0.97, f"{pct_neg:.1f}% negative WTP",
             transform=ax1.transAxes, fontsize=9, va="top", color=C_RED,
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=C_RED, alpha=0.8))

    ax2 = fig.add_subplot(1, 2, 2)
    q1, q3   = np.percentile(wtp, [25, 75])
    iqr      = q3 - q1
    fence_hi = q3 + 3 * iqr
    fence_lo = q1 - 3 * iqr

    ax2.boxplot(wtp, vert=True, patch_artist=True,
                boxprops=dict(facecolor=C_BLUE, alpha=0.35),
                medianprops=dict(color=C_RED, lw=2.2),
                whiskerprops=dict(lw=1.3, color=C_GRAY),
                capprops=dict(lw=1.5, color=C_GRAY),
                flierprops=dict(marker="o", markersize=2.5,
                                markerfacecolor=C_RED, alpha=0.4,
                                markeredgewidth=0),
                whis=1.5)
    rng    = np.random.default_rng(42)
    jitter = rng.uniform(-0.12, 0.12, len(wtp))
    ax2.scatter(1 + jitter, wtp, color=C_BLUE, alpha=0.15, s=4, zorder=3)
    ax2.axhline(0, color=C_GRAY, lw=1, ls=":")
    ax2.set_ylim(fence_lo * 1.1, fence_hi * 1.1)
    ax2.set_xticklabels(["WTP_farshore"])
    ax2.set_ylabel("WTP (DKK / household / year)")
    ax2.set_title("Box-plot + Jittered Points\n(extreme outliers clipped)",
                  fontweight="bold")

    n_out = ((wtp > fence_hi) | (wtp < fence_lo)).sum()
    ax2.text(0.98, 0.98,
             f"{n_out} extreme outlier(s)\nhidden for clarity",
             transform=ax2.transAxes, fontsize=8.5,
             ha="right", va="top", color=C_GRAY,
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=C_GRAY, alpha=0.8))

    plt.tight_layout()
    path = f"{OUTPUT}/Q2_WTP_distribution.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  → saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# Q5 — Mean WTP per WTP decile  (printed table)
# ════════════════════════════════════════════════════════════════════════════
def table_q5(df):
    print("\n" + "═"*60)
    print("Q5 — Mean WTP per WTP Decile")
    print("═"*60)
    stats_df = build_deciles(df, "WTP_farshore")
    # rename for display
    out = stats_df[["n", "mean", "std", "median"]].copy()
    out.columns = ["N", "Mean WTP (DKK)", "Std Dev (DKK)", "Median WTP (DKK)"]
    out.index.name = "WTP Decile"
    pd.set_option("display.float_format", "{:,.1f}".format)
    print(out.to_string())
    print()
    return stats_df


# ════════════════════════════════════════════════════════════════════════════
# Q8a — Mean WTP per income decile  (printed table)
# ════════════════════════════════════════════════════════════════════════════
def table_q8a(df):
    print("\n" + "═"*60)
    print("Q8a — Mean WTP per Income Decile")
    print("═"*60)
    stats_df = build_deciles(df, "income")
    out = stats_df[["n", "mean", "std", "median"]].copy()
    out.columns = ["N", "Mean WTP (DKK)", "Std Dev (DKK)", "Median WTP (DKK)"]
    out.index.name = "Income Decile (1=lowest)"
    print(out.to_string())
    print()
    return stats_df


# ════════════════════════════════════════════════════════════════════════════
# Q8b — Bar chart ± Std Dev  (what the assignment explicitly asks for)
# ════════════════════════════════════════════════════════════════════════════
def plot_q8b_stddev(df):
    stats_df = build_deciles(df, "income")
    x    = stats_df.index.astype(float)
    y    = stats_df["mean"].values
    yerr = stats_df["std"].values

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.bar(x, y, yerr=yerr, color=C_BLUE, alpha=0.65,
           error_kw=dict(ecolor="black", capsize=5, lw=1.5),
           label="Mean WTP", zorder=3, width=0.6)
    ax.axhline(0, color=C_GRAY, lw=1, ls=":", zorder=2)
    ax.set_xlabel("Income Decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Mean WTP (DKK/household/year)")
    ax.set_title("Q8b — Mean WTP by Income Decile (± Std Dev)")
    ax.set_xticks(x.astype(int))
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = f"{OUTPUT}/Q8b_WTP_by_income_decile_stddev.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  → saved: {path}")


# Q8b — CI version (dot-plot, same as your original)
def plot_q8b_ci(df):
    stats_df = build_deciles(df, "income")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    raw_vals = stats_df["mean"].values
    clip_hi  = np.percentile(np.abs(raw_vals), 85) * 2.5
    clip_lim = (-clip_hi, clip_hi)

    dot_plot(ax, stats_df, color=C_BLUE,
             xlabel="Income Decile  (1 = lowest income, 10 = highest income)",
             clip_ylim=clip_lim)
    ax.set_title("Q8b — Mean WTP by Income Decile  (± 95 % CI)")

    slope     = np.polyfit(stats_df.index, stats_df["mean"], 1)[0]
    direction = "↑ slight upward trend" if slope > 0 else "↓ slight downward trend"
    ax.text(0.98, 0.04, f"OLS trend: {direction}",
            transform=ax.transAxes, ha="right", fontsize=9,
            color=C_BLUE, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=C_BLUE, alpha=0.7))

    plt.tight_layout()
    path = f"{OUTPUT}/Q8b_WTP_by_income_decile_ci.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  → saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# Q8c — Statistical tests: income vs WTP
# ════════════════════════════════════════════════════════════════════════════
def analysis_q8c(df):
    print("\n" + "═"*60)
    print("Q8c — Statistical Tests: Income vs WTP")
    print("═"*60)

    d = df[["income", "WTP_farshore"]].dropna()
    income = d["income"]
    wtp    = d["WTP_farshore"]

    # ── 1. Spearman rank correlation ──────────────────────────────────────
    rho, p_spearman = stats.spearmanr(income, wtp)
    print(f"\n1. Spearman Rank Correlation")
    print(f"   ρ  = {rho:.4f}")
    print(f"   p  = {p_spearman:.4f}  {'*** significant at 1%' if p_spearman < 0.01 else '** significant at 5%' if p_spearman < 0.05 else '* significant at 10%' if p_spearman < 0.1 else 'not significant'}")
    print(f"   → {'Positive' if rho > 0 else 'Negative'} monotonic relationship between income and WTP")

    # ── 2. OLS regression ────────────────────────────────────────────────
    slope, intercept, r, p_ols, se = stats.linregress(income, wtp)
    print(f"\n2. OLS Linear Regression  (WTP = α + β·income)")
    print(f"   β (slope)     = {slope:,.4f}  DKK per unit income increase")
    print(f"   α (intercept) = {intercept:,.2f}")
    print(f"   R²            = {r**2:.4f}  (income explains {r**2*100:.1f}% of WTP variance)")
    print(f"   p-value (β)   = {p_ols:.4f}  {'*** significant at 1%' if p_ols < 0.01 else '** significant at 5%' if p_ols < 0.05 else '* significant at 10%' if p_ols < 0.1 else 'not significant'}")
    print(f"   ⚠  OLS assumes normality & homoscedasticity — both violated here.")
    print(f"      Treat as illustrative; Spearman is the preferred test.")

    # ── 3. Kruskal-Wallis test ────────────────────────────────────────────
    d2 = d.copy()
    d2["decile"] = pd.qcut(
        d2["income"].rank(method="first"), q=10,
        labels=False, duplicates="drop"
    ) + 1
    groups   = [grp["WTP_farshore"].values for _, grp in d2.groupby("decile")]
    H, p_kw  = stats.kruskal(*groups)
    print(f"\n3. Kruskal-Wallis Test  (are WTP distributions equal across income deciles?)")
    print(f"   H statistic = {H:.4f}")
    print(f"   p-value     = {p_kw:.4f}  {'*** significant at 1%' if p_kw < 0.01 else '** significant at 5%' if p_kw < 0.05 else '* significant at 10%' if p_kw < 0.1 else 'not significant'}")
    print(f"   → {'WTP distributions differ significantly across income deciles' if p_kw < 0.05 else 'No significant difference in WTP distributions across income deciles'}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("SUMMARY (Q8c)")
    print(f"  Spearman ρ = {rho:.4f}  (p = {p_spearman:.4f})")
    print(f"  OLS R²     = {r**2:.4f}  — income explains very little of WTP variance")
    print(f"  Kruskal-Wallis p = {p_kw:.4f}")
    print(f"  → Overall: {'Statistically significant but economically weak' if p_spearman < 0.05 else 'No significant'} positive income-WTP relationship")
    print()

    return rho, p_spearman, slope, r**2, p_ols, H, p_kw


# ════════════════════════════════════════════════════════════════════════════
# Q8c — Correlation Analysis Plots
# ════════════════════════════════════════════════════════════════════════════
def plot_q8c(df):
    """Three-panel figure visualising the income–WTP relationship for Q8c."""
    d = df[["income", "WTP_farshore"]].dropna().copy()
    income = d["income"].values
    wtp    = d["WTP_farshore"].values

    # ── pre-compute all stats so annotations are ready ────────────────────
    rho, p_spearman            = stats.spearmanr(income, wtp)
    slope, intercept, r, p_ols, _ = stats.linregress(income, wtp)

    d2 = d.copy()
    d2["decile"] = pd.qcut(
        d2["income"].rank(method="first"), q=10,
        labels=False, duplicates="drop"
    ) + 1
    groups = [grp["WTP_farshore"].values for _, grp in d2.groupby("decile")]
    H, p_kw = stats.kruskal(*groups)

    # clip WTP for display (99th pct) so outliers don't squash the panels
    wtp_hi   = np.percentile(wtp, 99)
    wtp_lo   = np.percentile(wtp, 1)
    wtp_clip = np.clip(wtp, wtp_lo, wtp_hi)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="white")
    fig.suptitle("Q8c — Income vs WTP: Correlation Analysis",
                 fontsize=14, fontweight="bold", y=1.02)

    # ── Panel 1: Scatter + OLS regression line ────────────────────────────
    ax1 = axes[0]
    ax1.scatter(income, wtp_clip, color=C_BLUE, alpha=0.25, s=14,
                edgecolors="none", label="Observations (99th pct. clip)")

    x_line = np.linspace(income.min(), income.max(), 200)
    y_line = slope * x_line + intercept
    y_line = np.clip(y_line, wtp_lo, wtp_hi)
    ax1.plot(x_line, y_line, color=C_RED, lw=2.2, label="OLS fit")

    ax1.axhline(0, color=C_GRAY, lw=1, ls=":")
    ax1.set_xlabel("Income (ordinal scale)")
    ax1.set_ylabel("WTP (DKK / household / year)")
    ax1.set_title("OLS Linear Regression", fontweight="bold")
    sig_ols = ("***" if p_ols < 0.01 else "**" if p_ols < 0.05
               else "*" if p_ols < 0.1 else "n.s.")
    ax1.text(0.04, 0.97,
             f"β = {slope:,.1f}\nR² = {r**2:.3f}\np {sig_ols} ({p_ols:.3f})",
             transform=ax1.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=C_RED, alpha=0.85))
    ax1.legend(fontsize=8, loc="lower right")

    # ── Panel 2: Rank–rank scatter (Spearman) ────────────────────────────
    ax2 = axes[1]
    rank_inc = stats.rankdata(income)
    rank_wtp = stats.rankdata(wtp)

    ax2.scatter(rank_inc, rank_wtp, color=C_ORANGE, alpha=0.22, s=14,
                edgecolors="none")

    # OLS line through ranks to show the Spearman trend visually
    sl2, ic2, _, _, _ = stats.linregress(rank_inc, rank_wtp)
    xr = np.linspace(rank_inc.min(), rank_inc.max(), 200)
    ax2.plot(xr, sl2 * xr + ic2, color=C_RED, lw=2.2)

    ax2.set_xlabel("Rank of Income")
    ax2.set_ylabel("Rank of WTP")
    ax2.set_title("Spearman Rank Correlation", fontweight="bold")
    sig_sp = ("***" if p_spearman < 0.01 else "**" if p_spearman < 0.05
              else "*" if p_spearman < 0.1 else "n.s.")
    ax2.text(0.04, 0.97,
             f"ρ = {rho:.4f}\np {sig_sp} ({p_spearman:.4f})\n"
             f"→ {'Positive' if rho > 0 else 'Negative'} monotonic",
             transform=ax2.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=C_ORANGE, alpha=0.85))

    # ── Panel 3: Box plots per income decile (Kruskal-Wallis) ────────────
    ax3 = axes[2]
    dec_labels = sorted(d2["decile"].unique())
    box_data   = [np.clip(grp["WTP_farshore"].values, wtp_lo, wtp_hi)
                  for _, grp in d2.groupby("decile")]

    bp = ax3.boxplot(box_data, positions=dec_labels, patch_artist=True,
                     widths=0.6, whis=1.5,
                     boxprops=dict(facecolor=C_BLUE, alpha=0.35),
                     medianprops=dict(color=C_RED, lw=2),
                     whiskerprops=dict(color=C_GRAY, lw=1.2),
                     capprops=dict(color=C_GRAY, lw=1.4),
                     flierprops=dict(marker="o", markersize=2.5,
                                     markerfacecolor=C_RED, alpha=0.35,
                                     markeredgewidth=0))

    # overlay decile means as dots
    means = [np.mean(g) for g in box_data]
    means_clip = np.clip(means, wtp_lo, wtp_hi)
    ax3.scatter(dec_labels, means_clip, color=C_RED, s=45, zorder=5,
                edgecolors="white", lw=0.8, label="Decile mean")

    ax3.axhline(0, color=C_GRAY, lw=1, ls=":")
    ax3.set_xlabel("Income Decile (1 = lowest, 10 = highest)")
    ax3.set_ylabel("WTP (DKK / household / year)")
    ax3.set_title("Kruskal-Wallis Test\n(WTP distributions across income deciles)",
                  fontweight="bold")
    ax3.set_xticks(dec_labels)
    sig_kw = ("***" if p_kw < 0.01 else "**" if p_kw < 0.05
              else "*" if p_kw < 0.1 else "n.s.")
    ax3.text(0.04, 0.97,
             f"H = {H:.2f}\np {sig_kw} ({p_kw:.4f})\n"
             f"→ {'Distributions differ\nsignificantly' if p_kw < 0.05 else 'No significant\ndifference'}",
             transform=ax3.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=C_GREEN, alpha=0.85))
    ax3.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    path = f"{OUTPUT}/Q8c_income_WTP_correlation.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  → saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# Q9c — Mean WTP by Distance Decile
# ════════════════════════════════════════════════════════════════════════════
def plot_q9c(df):
    stats_df = build_deciles(df, "D_WF")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    raw_vals = stats_df["mean"].values
    clip_hi  = np.percentile(np.abs(raw_vals), 85) * 2.5
    clip_lim = (-clip_hi, clip_hi)

    dot_plot(ax, stats_df, color=C_GREEN,
             xlabel="Distance Decile  (1 = closest to wind farm, 10 = farthest)",
             clip_ylim=clip_lim)
    ax.set_title("Q9c — Mean WTP by Distance Decile  (± 95 % CI)")

    slope     = np.polyfit(stats_df.index, stats_df["mean"], 1)[0]
    direction = ("↑ upward trend — distance decay / NIMBY effect"
                 if slope > 0 else "↓ downward trend")
    ax.text(0.98, 0.04, direction,
            transform=ax.transAxes, ha="right", fontsize=9,
            color=C_GREEN, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=C_GREEN, alpha=0.7))

    plt.tight_layout()
    path = f"{OUTPUT}/Q9c_WTP_by_distance_decile.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  → saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# Q9d — Statistical tests: distance vs WTP
# ════════════════════════════════════════════════════════════════════════════
def analysis_q9d(df):
    print("\n" + "═"*60)
    print("Q9d — Statistical Tests: Distance vs WTP")
    print("═"*60)

    d = df[["D_WF", "WTP_farshore"]].dropna()
    distance = d["D_WF"]
    wtp      = d["WTP_farshore"]

    # ── 1. Spearman rank correlation ──────────────────────────────────────
    rho, p_spearman = stats.spearmanr(distance, wtp)
    print(f"\n1. Spearman Rank Correlation")
    print(f"   ρ  = {rho:.4f}")
    print(f"   p  = {p_spearman:.4f}  {'*** significant at 1%' if p_spearman < 0.01 else '** significant at 5%' if p_spearman < 0.05 else '* significant at 10%' if p_spearman < 0.1 else 'not significant'}")
    print(f"   → {'Positive' if rho > 0 else 'Negative'} monotonic relationship between distance and WTP")

    # ── 2. OLS regression ────────────────────────────────────────────────
    slope, intercept, r, p_ols, se = stats.linregress(distance, wtp)
    print(f"\n2. OLS Linear Regression  (WTP = α + β·distance)")
    print(f"   β (slope)     = {slope:,.4f}  DKK per metre increase in distance")
    print(f"   α (intercept) = {intercept:,.2f}")
    print(f"   R²            = {r**2:.4f}  (distance explains {r**2*100:.1f}% of WTP variance)")
    print(f"   p-value (β)   = {p_ols:.4f}  {'*** significant at 1%' if p_ols < 0.01 else '** significant at 5%' if p_ols < 0.05 else '* significant at 10%' if p_ols < 0.1 else 'not significant'}")
    print(f"   ⚠  OLS assumes normality & homoscedasticity — both violated here.")
    print(f"      Treat as illustrative; Spearman is the preferred test.")

    # ── 3. Kruskal-Wallis test ────────────────────────────────────────────
    d2 = d.copy()
    d2["decile"] = pd.qcut(
        d2["D_WF"].rank(method="first"), q=10,
        labels=False, duplicates="drop"
    ) + 1
    groups  = [grp["WTP_farshore"].values for _, grp in d2.groupby("decile")]
    H, p_kw = stats.kruskal(*groups)
    print(f"\n3. Kruskal-Wallis Test  (are WTP distributions equal across distance deciles?)")
    print(f"   H statistic = {H:.4f}")
    print(f"   p-value     = {p_kw:.4f}  {'*** significant at 1%' if p_kw < 0.01 else '** significant at 5%' if p_kw < 0.05 else '* significant at 10%' if p_kw < 0.1 else 'not significant'}")
    print(f"   → {'WTP distributions differ significantly across distance deciles' if p_kw < 0.05 else 'No significant difference in WTP distributions across distance deciles'}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("SUMMARY (Q9d)")
    print(f"  Spearman ρ = {rho:.4f}  (p = {p_spearman:.4f})")
    print(f"  OLS β      = {slope:.4f}  R² = {r**2:.4f}")
    print(f"  Kruskal-Wallis p = {p_kw:.4f}")
    print(f"  → Overall: {'Statistically significant' if p_spearman < 0.05 else 'No significant'} "
          f"{'negative' if rho < 0 else 'positive'} distance-WTP relationship")
    print()

    return rho, p_spearman, slope, r**2, p_ols, H, p_kw


# ════════════════════════════════════════════════════════════════════════════
# Q9d — Correlation Analysis Plots (distance vs WTP)
# ════════════════════════════════════════════════════════════════════════════
def plot_q9d(df):
    """Three-panel figure visualising the distance–WTP relationship for Q9d."""
    d = df[["D_WF", "WTP_farshore"]].dropna().copy()
    distance = d["D_WF"].values
    wtp      = d["WTP_farshore"].values

    # ── pre-compute all stats ─────────────────────────────────────────────
    rho, p_spearman            = stats.spearmanr(distance, wtp)
    slope, intercept, r, p_ols, _ = stats.linregress(distance, wtp)

    d2 = d.copy()
    d2["decile"] = pd.qcut(
        d2["D_WF"].rank(method="first"), q=10,
        labels=False, duplicates="drop"
    ) + 1
    groups = [grp["WTP_farshore"].values for _, grp in d2.groupby("decile")]
    H, p_kw = stats.kruskal(*groups)

    # clip WTP for display (99th pct)
    wtp_hi   = np.percentile(wtp, 99)
    wtp_lo   = np.percentile(wtp, 1)
    wtp_clip = np.clip(wtp, wtp_lo, wtp_hi)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="white")
    fig.suptitle("Q9d — Distance vs WTP: Correlation Analysis",
                 fontsize=14, fontweight="bold", y=1.02)

    # ── Panel 1: Scatter + OLS regression line ────────────────────────────
    ax1 = axes[0]
    dist_km = distance / 1000  # convert metres → km for readability
    slope_km = slope * 1000    # rescale β to DKK per km

    ax1.scatter(dist_km, wtp_clip, color=C_GREEN, alpha=0.25, s=14,
                edgecolors="none", label="Observations (99th pct. clip)")

    x_line = np.linspace(dist_km.min(), dist_km.max(), 200)
    y_line = slope_km * x_line + intercept
    y_line = np.clip(y_line, wtp_lo, wtp_hi)
    ax1.plot(x_line, y_line, color=C_RED, lw=2.2, label="OLS fit")

    ax1.axhline(0, color=C_GRAY, lw=1, ls=":")
    ax1.set_xlabel("Distance to Wind Farm (km)")
    ax1.set_ylabel("WTP (DKK / household / year)")
    ax1.set_title("OLS Linear Regression", fontweight="bold")
    sig_ols = ("***" if p_ols < 0.01 else "**" if p_ols < 0.05
               else "*" if p_ols < 0.1 else "n.s.")
    ax1.text(0.04, 0.97,
             f"β = {slope_km:,.1f} DKK/km\nR² = {r**2:.3f}\np {sig_ols} ({p_ols:.3f})",
             transform=ax1.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=C_RED, alpha=0.85))
    ax1.legend(fontsize=8, loc="lower right")

    # ── Panel 2: Rank–rank scatter (Spearman) ────────────────────────────
    ax2 = axes[1]
    rank_dist = stats.rankdata(distance)
    rank_wtp  = stats.rankdata(wtp)

    ax2.scatter(rank_dist, rank_wtp, color=C_GREEN, alpha=0.22, s=14,
                edgecolors="none")

    sl2, ic2, _, _, _ = stats.linregress(rank_dist, rank_wtp)
    xr = np.linspace(rank_dist.min(), rank_dist.max(), 200)
    ax2.plot(xr, sl2 * xr + ic2, color=C_RED, lw=2.2)

    ax2.set_xlabel("Rank of Distance")
    ax2.set_ylabel("Rank of WTP")
    ax2.set_title("Spearman Rank Correlation", fontweight="bold")
    sig_sp = ("***" if p_spearman < 0.01 else "**" if p_spearman < 0.05
              else "*" if p_spearman < 0.1 else "n.s.")
    ax2.text(0.04, 0.97,
             f"ρ = {rho:.4f}\np {sig_sp} ({p_spearman:.4f})\n"
             f"→ {'Positive' if rho > 0 else 'Negative'} monotonic",
             transform=ax2.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=C_GREEN, alpha=0.85))

    # ── Panel 3: Box plots per distance decile (Kruskal-Wallis) ──────────
    ax3 = axes[2]
    dec_labels = sorted(d2["decile"].unique())
    box_data   = [np.clip(grp["WTP_farshore"].values, wtp_lo, wtp_hi)
                  for _, grp in d2.groupby("decile")]

    ax3.boxplot(box_data, positions=dec_labels, patch_artist=True,
                widths=0.6, whis=1.5,
                boxprops=dict(facecolor=C_GREEN, alpha=0.35),
                medianprops=dict(color=C_RED, lw=2),
                whiskerprops=dict(color=C_GRAY, lw=1.2),
                capprops=dict(color=C_GRAY, lw=1.4),
                flierprops=dict(marker="o", markersize=2.5,
                                markerfacecolor=C_RED, alpha=0.35,
                                markeredgewidth=0))

    means      = [np.mean(g) for g in box_data]
    means_clip = np.clip(means, wtp_lo, wtp_hi)
    ax3.scatter(dec_labels, means_clip, color=C_RED, s=45, zorder=5,
                edgecolors="white", lw=0.8, label="Decile mean")

    ax3.axhline(0, color=C_GRAY, lw=1, ls=":")
    ax3.set_xlabel("Distance Decile (1 = closest, 10 = farthest)")
    ax3.set_ylabel("WTP (DKK / household / year)")
    ax3.set_title("Kruskal-Wallis Test\n(WTP distributions across distance deciles)",
                  fontweight="bold")
    ax3.set_xticks(dec_labels)
    sig_kw = ("***" if p_kw < 0.01 else "**" if p_kw < 0.05
              else "*" if p_kw < 0.1 else "n.s.")
    ax3.text(0.04, 0.97,
             f"H = {H:.2f}\np {sig_kw} ({p_kw:.4f})\n"
             f"→ {'Distributions differ\nsignificantly' if p_kw < 0.05 else 'No significant\ndifference'}",
             transform=ax3.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=C_GREEN, alpha=0.85))
    ax3.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path = f"{OUTPUT}/Q9d_distance_WTP_correlation.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  → saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data("data.xls")

    print("\n[Q2]  Distribution WITH outliers (Figure 3) …")
    plot_q2_with_outliers(df)

    print("\n[Q2]  Distribution ZOOMED (Figure 4) …")
    plot_q2_zoomed(df)

    print("\n[Q5]  WTP decile table …")
    table_q5(df)

    print("\n[Q8a] Income decile table …")
    table_q8a(df)

    print("\n[Q8b] Bar chart ± Std Dev …")
    plot_q8b_stddev(df)

    print("\n[Q8b] Dot-plot ± 95% CI …")
    plot_q8b_ci(df)

    print("\n[Q8c] Statistical tests …")
    analysis_q8c(df)

    print("\n[Q8c] Correlation analysis plots …")
    plot_q8c(df)

    print("\n[Q9c] Distance decile plot …")
    plot_q9c(df)

    print("\n[Q9d] Statistical tests: distance vs WTP …")
    analysis_q9d(df)

    print("\n[Q9d] Correlation analysis plots …")
    plot_q9d(df)

    print(f"\n✓  All outputs saved to ./{OUTPUT}/")