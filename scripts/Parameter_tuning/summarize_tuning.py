import pandas as pd
from pathlib import Path
import json
import numpy as np

def summarize_tuning(results_dir="Experiments/Parameter_tuning/tuning_results_complete"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Path {results_path} does not exist.")
        return

    print(f"Scanning {results_path} for all individual run results...")
    
    all_runs = []
    
    for run_meta_path in sorted(results_path.glob("config_*/run_*/run_meta.json")):
        try:
            with open(run_meta_path, 'r') as f:
                meta = json.load(f)
            
            c_id = int(run_meta_path.parent.parent.name.split('_')[1])
            r_id = int(run_meta_path.parent.name.split('_')[1])
            
            config_data = meta.get("config", {})
            
            entry = {
                "config_id": c_id,
                "run_id": r_id,
                "status": meta.get("status", "unknown"),
                "test_rmse": meta.get("test_rmse", float("inf")),
                "val_loss": meta.get("final_val_loss", float("inf")),
                "training_time": meta.get("train_time_seconds", None),
                "error": meta.get("error", ""),
            }
            # Flatten config into columns
            for k, v in config_data.items():
                entry[k] = v
            all_runs.append(entry)
            
        except Exception:
            continue

    if not all_runs:
        print("No completed runs found at all.")
        return

    df_flat = pd.DataFrame(all_runs)
    df_flat = df_flat.sort_values(["config_id", "run_id"])
    flat_csv = results_path / "tuning_results.csv"
    df_flat.to_csv(flat_csv, index=False)
    print(f"Total individual runs collected: {len(df_flat)}")
    print(f"Flat summary saved to: {flat_csv}")

    df_success = df_flat[df_flat["status"].isin(["success", "completed"])].copy()
    
    if df_success.empty:
        print("No successful runs found to aggregate.")
        return

    grouped = df_success.groupby("config_id").agg({
        "test_rmse": ["mean", "std", "median", "max", "min", "count"],
        "val_loss": ["mean", "std"],
        "training_time": ["mean", "std"]
    })
    
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    grouped.rename(columns={"test_rmse_count": "n_runs"}, inplace=True)
    
  
    grouped["std_test_rmse"] = grouped["test_rmse_std"].fillna(grouped["test_rmse_mean"] * 0.5)
    grouped["mean_test_rmse"] = grouped["test_rmse_mean"]


    metric_cols = ["config_id", "test_rmse", "val_loss", "training_time", "status", "run_id", "error", "actual_steps_completed"]
    param_cols = [c for c in df_flat.columns if c not in metric_cols]
    
    config_params = df_flat[["config_id"] + param_cols].drop_duplicates("config_id")
    df_summary = pd.merge(grouped, config_params, on="config_id")
    
    df_summary = df_summary.sort_values("mean_test_rmse")
    
    priority = ["config_id", "mean_test_rmse", "std_test_rmse", "n_runs"]
    ordered_cols = priority + [c for c in df_summary.columns if c not in priority]
    df_summary = df_summary[ordered_cols]
    
    summary_csv = results_path / "tuning_summary.csv"
    df_summary.to_csv(summary_csv, index=False)


    print(df_summary.head(10).to_string(index=False))
 

if __name__ == "__main__":
    summarize_tuning()
