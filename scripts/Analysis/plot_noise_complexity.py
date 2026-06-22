
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "grid.color": "#aaaaaa",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": False,
    "figure.dpi": 150,
})


COMPLEXITY_ORDER  = ["Original", "Reactions_to_dahp_known", "Only_product_unknown"]
COMPLEXITY_LABELS = {
    "Original":                  "Low complexity\n(Original / Q1)",
    "Reactions_to_dahp_known":   "Medium complexity\n(DAHP reactions known)",
    "Only_product_unknown":      "High complexity\n(only product unknown)",
}
COMPLEXITY_SHORT = {
    "Original":                  "Low (Original)",
    "Reactions_to_dahp_known":   "Medium (Path to DAHP known)",
    "Only_product_unknown":      "High (Only product masked)",
}
COMPLEXITY_COLORS = {
    "Original":                "#2196F3",   # blue
    "Reactions_to_dahp_known": "#FF9800",   # orange
    "Only_product_unknown":    "#E91E63",   # pink
}
COMPLEXITY_MARKERS = {
    "Original":                "o",
    "Reactions_to_dahp_known": "s",
    "Only_product_unknown":    "^",
}

NOISE_LEVELS  = [0, 5, 10, 20]
STRAIN_SIZES  = [8, 24, 50, 100, 200, 300]
TS_POINTS     = [1, 3, 7, 14, 30]

BASE = "Experiments"

def parse_experiment_name(name):
    """Return (mask_name, ts_points, n_strains) or None."""
    m = re.match(r"pCA_(.+?)_(\d+)_points_(\d+)_strains", name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None


def _load_json_safe(path):
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None


METRICS = ["NRMSE_All_Species", "NRMSE_pCA_Final", "RMSE_pCA_Final"]


def load_all_noise_data():
    """
    Scan results.json files from noise and 0%-noise baseline experiments.
    Extracts NRMSE_All_Species and NRMSE_pCA_Final for each run.

    Returns a DataFrame with columns:
        noise_level, complexity, ts_points, n_strains,
        NRMSE_All_Species, NRMSE_pCA_Final
    """
    rows = []

    for noise in [5, 10, 20]:
        for complexity in COMPLEXITY_ORDER:
            base = Path(BASE) / f"Noise_{noise}" / complexity
            if not base.exists():
                print(f"  [WARN] Missing: {base}")
                continue
            for results_file in base.glob("**/results.json"):
                # path: .../exp_name/subset_i/seed_j/results.json
                exp_name = results_file.parts[-4]
                m = re.match(r"pCA_.+?_(\d+)_points_(\d+)_strains$", exp_name)
                if not m:
                    continue
                ts_pts, n_str = int(m.group(1)), int(m.group(2))
                data = _load_json_safe(results_file)
                if data is None:
                    continue
                row = {
                    "noise_level": noise,
                    "complexity":  complexity,
                    "ts_points":   ts_pts,
                    "n_strains":   n_str,
                }
                for metric in METRICS:
                    row[metric] = float(data[metric]) if metric in data and data[metric] is not None else np.nan
                rows.append(row)

    q1_base = Path(BASE) / "Question_1"
    if q1_base.exists():
        for results_file in q1_base.glob("**/results.json"):
            exp_name = results_file.parts[-4]
            m = re.match(r"pCA_hybrid_(\d+)_points_(\d+)_strains$", exp_name)
            if not m:
                continue
            ts_pts, n_str = int(m.group(1)), int(m.group(2))
            data = _load_json_safe(results_file)
            if data is None:
                continue
            row = {
                "noise_level": 0,
                "complexity":  "Original",
                "ts_points":   ts_pts,
                "n_strains":   n_str,
            }
            for metric in METRICS:
                row[metric] = float(data[metric]) if metric in data and data[metric] is not None else np.nan
            rows.append(row)
    else:
        print(f"  [WARN] Missing: {q1_base}")

    Q2_MASK_MAP = {
        "Reactions_to_dahp_known": "Reactions_to_dahp_known",
        "Only_product_unknown":    "Only_product_unknown",
    }
    q2_base = Path(BASE) / "Question_2"
    if q2_base.exists():
        for results_file in q2_base.glob("**/results.json"):
            exp_name = results_file.parts[-4]
            m = re.match(r"pCA_(.+?)_(\d+)_points_(\d+)_strains$", exp_name)
            if not m:
                continue
            mask, ts_pts, n_str = m.group(1), int(m.group(2)), int(m.group(3))
            if mask not in Q2_MASK_MAP:
                continue
            data = _load_json_safe(results_file)
            if data is None:
                continue
            row = {
                "noise_level": 0,
                "complexity":  Q2_MASK_MAP[mask],
                "ts_points":   ts_pts,
                "n_strains":   n_str,
            }
            for metric in METRICS:
                row[metric] = float(data[metric]) if metric in data and data[metric] is not None else np.nan
            rows.append(row)
    else:
        print(f"  [WARN] Missing: {q2_base}")

    return pd.DataFrame(rows)


def agg_rmse(df, groupby_cols, value_col="NRMSE_All_Species"):
    """Aggregate *value_col* by groupby_cols, returning mean, std, sem, median, q25, q75."""
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in DataFrame. "
                       f"Available columns: {df.columns.tolist()}")
    g = df.groupby(groupby_cols)[value_col]
    agg = g.agg(
        mean=lambda x: np.nanmean(x),
        std=lambda x: np.nanstd(x, ddof=1) if len(x) > 1 else 0.0,
        sem=lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x))) if np.sum(~np.isnan(x)) > 1 else 0.0,
        n=lambda x: np.sum(~np.isnan(x)),
        median=lambda x: np.nanmedian(x),
        q25=lambda x: np.nanpercentile(x, 25),
        q75=lambda x: np.nanpercentile(x, 75),
    ).reset_index()
    return agg


def figure_strains_vs_complexity_grid(
        df,
        out_path,
        metric_col="test_rmse",
        metric_label="NRMSE",
        title_suffix="(all metabolites, all timepoints)",
        y_lim=None,
):
    """
    Full grid: rows = ts_points (1,3,7,14,30), cols = noise (0,5,10,20).
    Each cell: x = n_strains (500 excluded), y = mean metric, colour = complexity.
    Shaded band = mean ± 1 std.

    Parameters
    ----------
    metric_col    : column name in *df* to aggregate (e.g. 'test_rmse',
                    'test_rmse_pca').
    metric_label  : y-axis / colour-bar label string.
    title_suffix  : appended to the figure suptitle to distinguish variants.
    """
    # Exclude the 500-strain condition as noise experiments only go up to 300 strains
    df = df[df["n_strains"] != 500].copy()

    nrows = len(TS_POINTS)
    ncols = len(NOISE_LEVELS)

    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 14),
                             sharey=True, sharex=True)
    fig.subplots_adjust(hspace=0.22, wspace=0.10,
                        top=0.93, bottom=0.07, left=0.10, right=0.97)

    agg = agg_rmse(df, ["complexity", "noise_level", "ts_points", "n_strains"],
                   value_col=metric_col)

    row_ylims = {}
    for ts in TS_POINTS:
        row_vals = agg[agg["ts_points"] == ts]["mean"] + agg[agg["ts_points"] == ts]["std"]
        row_max = float(np.nanpercentile(row_vals, 97)) if len(row_vals) > 0 else 1.0
        row_min = float(np.nanpercentile(agg[agg["ts_points"] == ts]["mean"] - agg[agg["ts_points"] == ts]["std"], 2))
        row_min = max(row_min, 0.0)
        row_ylims[ts] = (row_min, row_max * 1.05)

    for row_idx, ts in enumerate(TS_POINTS):
        for col_idx, noise in enumerate(NOISE_LEVELS):
            ax = axes[row_idx, col_idx]
            sub = agg[(agg["ts_points"] == ts) & (agg["noise_level"] == noise)]

            for cpx in COMPLEXITY_ORDER:
                c_sub = sub[sub["complexity"] == cpx].sort_values("n_strains")
                if c_sub.empty:
                    continue
                col = COMPLEXITY_COLORS[cpx]
                mk  = COMPLEXITY_MARKERS[cpx]

                ax.plot(c_sub["n_strains"], c_sub["mean"],
                        color=col, marker=mk, markersize=4.5,
                        linewidth=1.5, zorder=3)
                ax.fill_between(c_sub["n_strains"],
                                c_sub["mean"] - c_sub["std"],
                                c_sub["mean"] + c_sub["std"],
                                color=col, alpha=0.12, zorder=2)

            # Clip Y to row-specific limits to prevent outliers squashing the view
            if y_lim:
                ax.set_ylim(y_lim[0], y_lim[1])



            ax.set_xticks([8, 50, 100, 200, 300])
            ax.tick_params(labelsize=7.5)
            ax.grid(axis="y")

            if row_idx == 0:
                ax.set_title(f"Noise: {noise}%", fontsize=10,
                             fontweight="bold", pad=6)

            if col_idx == 0:
                ts_lbl = {1: "1 timepoint", 3: "3 timepoints", 7: "7 timepoints",
                          14: "14 timepoints", 30: "30 timepoints"}
                ax.set_ylabel(f"{ts_lbl[ts]}\n{metric_label}", fontsize=8.5)

            if row_idx == nrows - 1:
                ax.set_xlabel("Strains", fontsize=8.5)

    legend_handles = [
        Line2D([0], [0], color=COMPLEXITY_COLORS[c], marker=COMPLEXITY_MARKERS[c],
               linewidth=1.8, markersize=6, label=COMPLEXITY_SHORT[c])
        for c in COMPLEXITY_ORDER
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0),
               fontsize=9, title="Underlying Mechanistic Complexity",
               title_fontsize=9.5, frameon=True,
               fancybox=False, edgecolor="#cccccc")


    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    out_dir = os.path.join("Experiments", "noise_complexity_figures")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data from all noise experiment trackers...")
    df = load_all_noise_data()

    if df.empty:
        print("ERROR: No data loaded. Check that Experiments_actual/Noise_*/*/experiment_tracker.csv exist.")
        return

    print(f"  Loaded {len(df):,} experiment runs across "
          f"{df['noise_level'].nunique()} noise levels, "
          f"{df['complexity'].nunique()} model complexities.")
    


    figure_strains_vs_complexity_grid(
        df,
        os.path.join(out_dir, "noise_grid_nrmse_all.png"),
        metric_col="NRMSE_All_Species",
        metric_label="NRMSE",
        title_suffix="(all metabolites, all timepoints)",
    )

    figure_strains_vs_complexity_grid(
        df,
        os.path.join(out_dir, "noise_grid_rmse_final_pca.png"),
        metric_col="RMSE_pCA_Final",
        metric_label="RMSE (p-CA final)",
        title_suffix="(final pCA concentration)",
        y_lim=(0.0, 0.003)
    )

    pca_col = "NRMSE_pCA_Final"
    if pca_col in df.columns:
        figure_strains_vs_complexity_grid(
            df,
            os.path.join(out_dir, "noise_grid_nrmse_pca_final.png"),
            metric_col=pca_col,
            metric_label="NRMSE (p-CA final)",
            title_suffix="(final pCA concentration point only)",
        )
    else:
        print(f"  [SKIP] Figure 4b: column '{pca_col}' not found in loaded data.")

    
  

    print(f"\nDone. All figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
