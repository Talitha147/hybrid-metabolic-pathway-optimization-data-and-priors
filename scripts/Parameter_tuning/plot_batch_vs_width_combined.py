import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "Experiments" / "Parameter_tuning" / "batch_size_vs_width"
OUTPUT_DIR = PROJECT_ROOT / "Experiments" / "Comparison_Summaries" / "batch_vs_width"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_folder(folder: Path) -> pd.DataFrame:
    records = []
    for config_dir in sorted(folder.glob("config_*")):
        if not config_dir.is_dir():
            continue
        for run_dir in sorted(config_dir.glob("run_*")):
            meta_path = run_dir / "run_meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            cfg = meta.get("config", {})
            status = meta.get("status", "unknown")
            if status not in ("success", "recovered", "checkpoint_recovered"):
                continue
            records.append({
                "batch_size": cfg.get("batch_size"),
                "width_NODE": cfg.get("width_NODE"),
                "test_rmse": meta.get("test_rmse", float("nan")),
                "train_time": meta.get("train_time_seconds", float("nan")),
            })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _draw_heatmap(ax, pivot: pd.DataFrame, fmt: str, cmap: str,
                  vmin, vmax, cbar_label: str):
 
    im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    # Ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Axis labels
    ax.set_xlabel("Batch size", fontsize=10)
    ax.set_ylabel("Neural width", fontsize=10)

    # Annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                txt = "N/A"
                color = "black"
            else:
                txt = f"{val:{fmt}}"
            
                norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = "white" if norm_val > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return im


def plot_combined(strain_label: str, summary: pd.DataFrame,
                  global_rmse_min, global_rmse_max,
                  global_time_min, global_time_max):
   

    agg = summary.groupby(["width_NODE", "batch_size"]).agg(
        test_rmse=("test_rmse", "mean"),
        train_time=("train_time", "mean"),
    ).reset_index()

    def make_pivot(col):
        piv = agg.pivot(index="width_NODE", columns="batch_size", values=col)
        piv = piv.sort_index(ascending=False)          # larger width at top
        piv = piv.reindex(sorted(piv.columns), axis=1) # smaller batch left
        return piv

    pivot_rmse = make_pivot("test_rmse")
    pivot_time = make_pivot("train_time")

    fig, (ax_rmse, ax_time) = plt.subplots(1, 2, figsize=(12, 5))

    _draw_heatmap(ax_rmse, pivot_rmse,
                  fmt=".4f", cmap="viridis",
                  vmin=global_rmse_min, vmax=global_rmse_max,
                  cbar_label="Mean RMSE")

    _draw_heatmap(ax_time, pivot_time,
                  fmt=".0f", cmap="magma",
                  vmin=global_time_min, vmax=global_time_max,
                  cbar_label="Mean train time (s)")

    plt.tight_layout()

    out_path = OUTPUT_DIR / f"heatmap_combined_{strain_label}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def main():
    if not EXPERIMENTS_DIR.exists():
        print(f"Directory not found: {EXPERIMENTS_DIR}")
        return

    strain_folders = sorted(
        [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()],
        key=lambda d: int(d.name.split("_")[0]) if d.name.split("_")[0].isdigit() else 0
    )

    if not strain_folders:
        print("No strain subfolders found.")
        return

    print(f"Found {len(strain_folders)} strain folders: {[f.name for f in strain_folders]}")

    # Load all data first to compute global scales
    all_data = {}
    for folder in strain_folders:
        df = load_folder(folder)
        if not df.empty:
            all_data[folder] = df
        else:
            print(f"  No successful runs in {folder.name}, skipping.")

    if not all_data:
        print("No data found.")
        return

    combined = pd.concat(all_data.values())
    global_rmse_min = combined["test_rmse"].min()
    global_rmse_max = combined["test_rmse"].max()
    global_time_min = combined["train_time"].min()
    global_time_max = combined["train_time"].max()

    print(f"\nGlobal RMSE range:  {global_rmse_min:.4f} – {global_rmse_max:.4f}")
    print(f"Global time range:  {global_time_min:.1f} – {global_time_max:.1f} s")
    print()

    for folder, df in all_data.items():
        strain_label = folder.name   # e.g. "8_strains"
        print(f"Plotting {strain_label}...")
        plot_combined(strain_label, df,
                      global_rmse_min, global_rmse_max,
                      global_time_min, global_time_max)

    print(f"\nDone. Figures saved in:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
