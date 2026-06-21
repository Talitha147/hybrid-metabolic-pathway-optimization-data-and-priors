import os
import sys
import jax.numpy as jnp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import gc
import jax


script_path = Path(__file__).parent.absolute()
project_root = script_path.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from scripts.models.hybrid_model import HybridModel
from scripts.data_generation import load_data_from_csvs

def calculate_rmse(y_true, y_pred):
    mask = jnp.isfinite(y_true) & jnp.isfinite(y_pred)
    if not jnp.any(mask):
        return float('inf')
    diff = y_true - y_pred
    safe_diff = jnp.where(mask, diff, 0.0)
    # Average only over valid points
    rmse = jnp.sqrt(jnp.sum(jnp.square(safe_diff)) / jnp.sum(mask))
    return float(rmse)



def plot_heatmap(df, x_param, y_param, value_col, title, filename, cmap="viridis", fmt=".2f", vmin=None, vmax=None):
    """
    Plots a heatmap from a DataFrame.
    """
    pivot = df.pivot(index=y_param, columns=x_param, values=value_col)
    pivot = pivot.sort_index(ascending=False) # Higher y at top
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax)
    

    plt.colorbar(im, ax=ax)
    
    # Set labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(title)
    
   
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text = ax.text(j, i, f"{val:{fmt[1:]}}" if not np.isnan(val) else "N/A",
                           ha="center", va="center", color="w" if im.norm(val) > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def analyze_2d_folder(folder_path, sbml_model, ts, ys_total, params_total, pCA_index):
    folder_path = Path(folder_path)
    print(f"\nAnalyzing folder: {folder_path}")
    
    all_runs_data = []
    
    for config_dir in sorted(folder_path.glob("config_*")):
        # Skip if not a directory
        if not config_dir.is_dir(): continue
        
        for run_dir in sorted(config_dir.glob("run_*")):
            if not run_dir.is_dir(): continue
            
            meta_path = run_dir / "run_meta.json"
            if not meta_path.exists():
                continue
                
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            status = meta.get("status")
            test_rmse = meta.get("test_rmse", float('nan'))
            train_time = meta.get("train_time_seconds", float('nan'))
            config = meta.get("config", {})
            
            batch_size = config.get("batch_size")
            width_node = config.get("width_NODE")
            
            
            pred_file = None
            for pf in ["test_preds_final.npy", "test_preds_checkpoint_recovered.npy", "test_preds_recovered.npy"]:
                if (run_dir / pf).exists():
                    pred_file = run_dir / pf
                    break
            
            if pred_file:
                try:
                    test_preds = jnp.load(pred_file)
                    test_indices = meta.get("test_indices", [])
                    if test_indices:
                        test_idx = np.array(test_indices)
                        ys_test = ys_total[test_idx]
                        if test_preds.shape[1] != ys_test.shape[1]:
                            n_preds = test_preds.shape[1]
                            sub_indices = jnp.linspace(0, ys_test.shape[1] - 1, n_preds).astype(int)
                            ys_test = ys_test[:, sub_indices, :]
                        test_rmse = calculate_rmse(ys_test, test_preds)
                except Exception:
                    pass

            all_runs_data.append({
                "batch_size": batch_size,
                "width_NODE": width_node,
                "status": status,
                "test_rmse": test_rmse,
                "train_time": train_time
            })

    if not all_runs_data:
        print(f"No run data found in {folder_path}")
        return None

    df = pd.DataFrame(all_runs_data)
    
    # Save combined results
    df.to_csv(folder_path / "all_runs_analysis.csv", index=False)
    
    # Aggregate by (batch_size, width_NODE)
    df['is_success'] = df['status'].apply(lambda x: 1 if x in ["success", "recovered", "checkpoint_recovered"] else 0)
    
    summary = df.groupby(['batch_size', 'width_NODE']).agg({
        'test_rmse': 'mean',
        'train_time': 'mean',
        'is_success': 'mean'
    }).reset_index()
    
    summary.rename(columns={'is_success': 'success_rate'}, inplace=True)
    summary.to_csv(folder_path / "tuning_heatmap_summary.csv", index=False)

    return summary

def main():
    MODEL_PATH = 'models/pCA_model_changed_S.xml'
    DATA_DIR = 'data'
    EXPERIMENTS_DIR = Path('Experiments_actual/Parameter_tuning/batch_size_vs_width')
    
    if not EXPERIMENTS_DIR.exists():
        # Try relative to project root
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent
        EXPERIMENTS_DIR = project_root / 'Experiments_actual' / 'Parameter_tuning' / 'batch_size_vs_width'
        
    if not EXPERIMENTS_DIR.exists():
        print(f"Experiments directory not found: {EXPERIMENTS_DIR}")
        return

    print(f"Loading SBML model from: {MODEL_PATH}")
    if not Path(MODEL_PATH).exists():
        MODEL_PATH = str(project_root / MODEL_PATH)
        
    sbml_model = SBMLModel(MODEL_PATH)
    pCA_index = sbml_model.species_names.index('pcoumaric_acid') if 'pcoumaric_acid' in sbml_model.species_names else 0
    print(f"pCA index identified: {pCA_index}")
    
    print(f"Loading data from: {DATA_DIR}")
    if not Path(DATA_DIR).exists():
        DATA_DIR = str(project_root / DATA_DIR)
        
    ts, ys_total, params_total, _ = load_data_from_csvs(
        csv_dir=DATA_DIR,
        metabolites=sbml_model.species_names,
        model_name="pCA_model",
    )
    
    all_summaries = []
    for folder in sorted(list(EXPERIMENTS_DIR.iterdir())):
        if folder.is_dir() and not folder.name.startswith("."):
            summary = analyze_2d_folder(folder, sbml_model, ts, ys_total, params_total, pCA_index)
            if summary is not None:
                all_summaries.append((summary, folder))
    
    if not all_summaries:
        print("No summaries to plot.")
        return

    # Calculate global bounds for consistent scaling
    all_data_concat = pd.concat([s for s, f in all_summaries])
    
    global_rmse_min = all_data_concat['test_rmse'].min()
    global_rmse_max = all_data_concat['test_rmse'].max()
    global_time_min = all_data_concat['train_time'].min()
    global_time_max = all_data_concat['train_time'].max()
    
    print(f"\nGlobal Scales:")
    print(f"  RMSE: {global_rmse_min:.4f} to {global_rmse_max:.4f}")
    print(f"  Time: {global_time_min:.1f} to {global_time_max:.1f}")


    for summary, folder_path in all_summaries:
        # Plot Heatmaps
        plot_heatmap(summary, "batch_size", "width_NODE", "test_rmse", 
                     f"Mean Test RMSE - {folder_path.name}", folder_path / "heatmap_rmse.png", 
                     fmt=".4f", vmin=global_rmse_min, vmax=global_rmse_max)
        
        plot_heatmap(summary, "batch_size", "width_NODE", "train_time", 
                     f"Mean Training Time (s) - {folder_path.name}", folder_path / "heatmap_time.png", 
                     fmt=".1f", cmap="magma", vmin=global_time_min, vmax=global_time_max)
        
        plot_heatmap(summary, "batch_size", "width_NODE", "success_rate", 
                     f"Success Rate - {folder_path.name}", folder_path / "heatmap_success.png", 
                     fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)

        print(f"Updated plots for {folder_path.name} using global scales.")

if __name__ == "__main__":
    main()
