"""Figure functions with one consistent, restrained visual style.

Colors come from a validated colorblind-safe categorical order; a single
series always uses slot 1, ordered magnitudes use a one-hue blue ramp, and
risk-level colors are reserved for the risk matrix only.
"""

from __future__ import annotations

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#ffffff"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
            "savefig.dpi": 200,
        }
    )


def _clean_x_grid(ax):
    ax.grid(axis="x", visible=False)


def monthly_rate_plot(counts: pd.DataFrame, ci: pd.DataFrame, all_months: list[str]):
    """Incidents per 30 days by month with exact 95% CIs; uncovered months marked."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(all_months))
    covered = [m for m in all_months if m in counts.index]
    for m in covered:
        i = all_months.index(m)
        r = counts.loc[m, "n"] / counts.loc[m, "days"] * 30
        ax.bar(i, r, width=0.62, color=SERIES[0], zorder=2)
        ax.errorbar(
            i,
            r,
            yerr=[[r - ci.loc[m, "lo"]], [ci.loc[m, "hi"] - r]],
            color=TEXT,
            lw=1.1,
            capsize=3,
            zorder=3,
        )
        ax.text(
            i,
            1.0,
            f"n={counts.loc[m, 'n']}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="white",
            zorder=4,
        )
    top = ax.get_ylim()[1]
    for i, m in enumerate(all_months):
        if m not in covered:
            ax.axvspan(i - 0.31, i + 0.31, color="#f1f0ed", zorder=1)
            ax.text(
                i, top * 0.45, "not\ncovered", ha="center", va="center", fontsize=8, color=MUTED
            )
    ax.set_xticks(x, [pd.Period(m).strftime("%b") for m in all_months])
    ax.set_ylabel("Incidents per 30 days")
    ax.set_title("Monthly incident rate with exact 95% intervals")
    _clean_x_grid(ax)
    fig.tight_layout()
    return fig, ax


def duration_by_group_plot(
    df: pd.DataFrame, group: str, order: list, labels: list[str], title: str, rng_seed: int = 0
):
    """Log-scale strip + box plot of durations by group."""
    rng = np.random.default_rng(rng_seed)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    data = [df.loc[df[group] == g, "duration_min"].dropna().values for g in order]
    ax.boxplot(
        data,
        positions=range(len(order)),
        widths=0.5,
        showfliers=False,
        medianprops=dict(color=TEXT, lw=1.6),
        boxprops=dict(color=TEXT_2, lw=1),
        whiskerprops=dict(color=TEXT_2, lw=1),
        capprops=dict(color=TEXT_2, lw=1),
    )
    for i, d in enumerate(data):
        ax.scatter(
            i + rng.uniform(-0.18, 0.18, len(d)),
            d,
            s=16,
            color=SERIES[0],
            alpha=0.65,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_yticks(
        [1, 5, 10, 30, 60, 120, 300, 600], ["1", "5", "10", "30", "60", "120", "300", "600"]
    )
    ax.set_xticks(range(len(order)), labels)
    ax.set_ylabel("Duration (minutes, log scale)")
    ax.set_title(title)
    _clean_x_grid(ax)
    fig.tight_layout()
    return fig, ax


def duration_distribution_plot(minutes: pd.Series):
    """Raw and log-scale histograms side by side, showing the log-normal shape."""
    x = minutes.dropna().values
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(x, bins=30, color=SERIES[0], edgecolor="white", linewidth=1)
    axes[0].set_xlabel("Duration (minutes)")
    axes[0].set_ylabel("Incidents")
    axes[0].set_title("Raw scale: strongly right-skewed")
    axes[1].hist(np.log10(x), bins=16, color=SERIES[0], edgecolor="white", linewidth=1)
    axes[1].set_xticks(np.log10([1, 3, 10, 30, 100, 300]), ["1", "3", "10", "30", "100", "300"])
    axes[1].set_xlabel("Duration (minutes, log scale)")
    axes[1].set_title("Log scale: approximately normal")
    for ax in axes:
        _clean_x_grid(ax)
    fig.tight_layout()
    return fig, axes


def hazard_rate_plot(freq: pd.DataFrame, labels: dict[str, str]):
    """Posterior mean rate per 30 days with 95% credible intervals, one row per hazard."""
    f = freq[freq["hazard"] != "All"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(f))
    ax.hlines(y, f["cri_lo_30d"], f["cri_hi_30d"], color=SERIES[0], lw=2, zorder=2)
    ax.scatter(f["post_mean_30d"], y, s=48, color=SERIES[0], edgecolor="white", lw=1.5, zorder=3)
    for yi, (_, r) in zip(y, f.iterrows(), strict=True):
        ax.text(r["cri_hi_30d"] + 0.25, yi, f"n={r['n']}", va="center", fontsize=8.5, color=TEXT_2)
    ax.set_yticks(y, [labels[h] for h in f["hazard"]])
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20], ["0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("Incidents per 30 days (posterior mean, 95% credible interval, log scale)")
    ax.set_title("Estimated frequency by hazard")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig, ax


def hazard_impact_heatmap(table: pd.DataFrame, row_labels: list[str], col_labels: list[str]):
    """Counts of incidents by hazard (rows) and impact level (columns)."""
    fig, ax = plt.subplots(figsize=(8, 3.9))
    vals = table.values
    cmap = mpl.colors.ListedColormap(["#f6f6f4", *BLUE_RAMP])
    bounds = [0, 0.5, 2.5, 5.5, 10.5, 20.5, 35.5, 50.5, 100]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if v:
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color="white" if v > 10 else TEXT,
                )
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xticks(np.arange(-0.5, vals.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, vals.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Impact level (incident-level score)")
    ax.set_title("Incidents by hazard and impact level")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig, ax


def grouped_share_plot(shares: pd.DataFrame, labels: list[str], title: str, ylabel: str):
    """Grouped bars: one group per row of ``shares``, one bar per column (<= 3 columns)."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    k = shares.shape[1]
    width = 0.8 / k
    x = np.arange(len(shares))
    for j, col in enumerate(shares.columns):
        ax.bar(
            x + (j - (k - 1) / 2) * width,
            shares[col].values,
            width=width * 0.92,
            color=SERIES[j],
            label=str(col),
            zorder=2,
        )
    ax.set_xticks(x, [textwrap.fill(lab, 12, break_long_words=False) for lab in labels])
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title(title)
    ax.legend(loc="upper left")
    _clean_x_grid(ax)
    fig.tight_layout()
    return fig, ax


def count_bar_plot(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str = "Incidents",
    horizontal: bool = False,
    figsize=(8, 3.6),
):
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        s = series.iloc[::-1]
        ax.barh(range(len(s)), s.values, color=SERIES[0], height=0.62, zorder=2)
        ax.set_yticks(range(len(s)), s.index)
        ax.set_xlabel(ylabel)
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(range(len(series)), series.values, color=SERIES[0], width=0.7, zorder=2)
        ax.set_xticks(range(len(series)), series.index)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _clean_x_grid(ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def classifier_comparison_plot(summary: pd.DataFrame):
    """Macro-F1 by method: CV mean +/- 1 SD for learned models, point values otherwise."""
    s = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(s))
    has_sd = s["macro_F1_sd"].notna()
    ax.hlines(
        y[has_sd],
        (s["macro_F1"] - s["macro_F1_sd"])[has_sd],
        (s["macro_F1"] + s["macro_F1_sd"])[has_sd],
        color=SERIES[0],
        lw=2,
    )
    ax.scatter(s["macro_F1"], y, s=50, color=SERIES[0], edgecolor="white", lw=1.5, zorder=3)
    for yi, v in zip(y, s["macro_F1"], strict=True):
        ax.text(v, yi + 0.28, f"{v:.2f}", ha="center", fontsize=8.5, color=TEXT_2)
    ax.set_yticks(y, s["method"])
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.6, len(s) - 0.25)
    ax.set_xlabel("Macro-averaged F1 against manual audit labels")
    ax.set_title("Narrative classifier comparison")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig, ax
