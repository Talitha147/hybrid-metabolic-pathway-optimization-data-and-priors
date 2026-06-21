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


# Add project root to sys.path
script_path = Path(__file__).parent.absolute()
project_root = script_path.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from models.hybrid_model import HybridModel
from data_generation import load_data_from_csvs

def calculate_rmse(y_true, y_pred):
    mask = jnp.isfinite(y_true) & jnp.isfinite(y_pred)
    if not jnp.any(mask):
        return float('inf')
    diff = y_true - y_pred
    safe_diff = jnp.where(mask, diff, 0.0)
    # Average only over valid points
    rmse = jnp.sqrt(jnp.sum(jnp.square(safe_diff)) / jnp.sum(mask))
    return float(rmse)

def get_divergence_step(run_dir):
    loss_file = run_dir / "loss_history.csv"
    if not loss_file.exists():
        return None
    try:
        df = pd.read_csv(loss_file)
        if df.empty or "train_loss" not in df.columns:
            return None
            
        min_loss = df["train_loss"].min()
        limit = max(min_loss * 50, 1.0)
        
        for i, row in df.iterrows():
            loss = row["train_loss"]
            if pd.isna(loss) or loss > limit:
                return i
        return None
    except Exception:
        return None

def get_latest_checkpoint(run_dir, diverged_step=None):
    checkpoints = sorted(list(run_dir.glob("step_*")), key=lambda x: int(x.name.split("_")[1]), reverse=True)
    if not checkpoints:
        return None
        
    # If we diverged, the checkpoint at that exact step likely contains NaNs
    # Skip it and find the next one
    for cp in checkpoints:
        step = int(cp.name.split("_")[1])
        if diverged_step is not None and step >= diverged_step:
            continue
        return cp
    
    return checkpoints[0]

def analyze_parameter_folder(folder_path, sbml_model, ts, ys_total, params_total, pCA_index, parameter_name=None):
    folder_path = Path(folder_path)

    if not parameter_name:
        parameter_name = folder_path.name

    print(f"\nAnalyzing folder: {folder_path}")
    
    all_runs_data = []
    
    for config_dir in sorted(folder_path.glob("config_*")):
        config_id = int(config_dir.name.split("_")[1])
        
        for run_dir in sorted(config_dir.glob("run_*")):
            run_id = int(run_dir.name.split("_")[1])
            meta_path = run_dir / "run_meta.json"
            
            if not meta_path.exists():
                continue
                
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            status = meta.get("status")
            stopping_condition = meta.get("stopping_condition", "unknown")
            test_rmse = meta.get("test_rmse", float('nan'))
            train_time = meta.get("train_time_seconds", float('nan'))
            config = meta.get("config", {})
            param_value = config.get(parameter_name, "unknown")
            
            model_source = "finished"
            
            # Priority: final > checkpoint > recovered
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
                        
                        # Handle timepoint mismatch (e.g. 8 vs 40 points)
                        if test_preds.shape[1] != ys_test.shape[1]:
                            n_preds = test_preds.shape[1]
                            # Recreate indices used for subsampling during training
                            sub_indices = jnp.linspace(0, ys_test.shape[1] - 1, n_preds).astype(int)
                            ys_test = ys_test[:, sub_indices, :]
                        
                        test_rmse = calculate_rmse(ys_test, test_preds)
                        if "checkpoint" in pred_file.name:
                            status = "checkpoint_recovered"
                            model_source = "checkpoint_preds"
                        elif "recovered" in pred_file.name:
                            status = "recovered"
                            model_source = "recovered_preds"
                except Exception as e:
                    print(f"  Error loading cached predictions {pred_file}: {e}")
                    pred_file = None

            if not pred_file:
                # Determine precise step where loss jumped using loss_history.csv
                diverged_step = None
                if stopping_condition == "nan_diverged" or test_rmse == 0.0 or jnp.isnan(test_rmse) or test_rmse > 1e5:
                    diverged_step = get_divergence_step(run_dir)
                if diverged_step is None and stopping_condition == "nan_diverged":
                    diverged_step = meta.get("actual_steps_completed")
                
                should_recover = (stopping_condition == "nan_diverged") or \
                                 (test_rmse == 0.0 and stopping_condition not in ["completed", "val_loss_stagnation"]) or \
                                 jnp.isnan(test_rmse) or (test_rmse > 1e5)
                
                if should_recover:
                    latest_cp = get_latest_checkpoint(run_dir, diverged_step=diverged_step)
                    if latest_cp:
                        print(f"  Attempting recovery for Config {config_id} Run {run_id} (Condition: {stopping_condition}) from {latest_cp.name}")
                        try:
                            # Load model from checkpoint
                            model = HybridModel.load(latest_cp)
                            
                            test_indices = meta.get("test_indices", [])
                            if test_indices:
                                test_idx = np.array(test_indices)
                                ys_test = ys_total[test_idx]
                                params_test = params_total[test_idx]
                                ts_test = ts
                                
                                test_preds = []
                                for idx_t in range(len(ys_test)):
                                    p_t = params_test[idx_t]
                                    y0_t = ys_test[idx_t, 0, :]
                                    y_p = model(ts_test, y0_t, p_t)
                                    test_preds.append(y_p)
                                
                                test_preds = jnp.stack(test_preds)
                                test_rmse = calculate_rmse(ys_test, test_preds)
                                status = "recovered"
                                model_source = latest_cp.name
                                # Save recovered predictions for next time
                                jnp.save(run_dir / "test_preds_recovered.npy", test_preds)
                                print(f"    Recovered RMSE: {test_rmse:.6f} (Saved to test_preds_recovered.npy)")
                            
                            del model
                        except Exception as e:
                            print(f"    Failed to recover: {e}")
            
            all_runs_data.append({
                "config_id": config_id,
                "run_id": run_id,
                "parameter": parameter_name,
                "value": param_value,
                "status": status,
                "stopping_condition": stopping_condition,
                "test_rmse": test_rmse,
                "train_time": train_time,
                "model_source": model_source
            })

    if not all_runs_data:
        return None

    # Consolidated resource cleanup after main analysis loop
    gc.collect()
    try:
        jax.clear_caches()
    except:
        pass

    df = pd.DataFrame(all_runs_data)
    
    # Save plotting summary
    plotting_summary_path = folder_path / "plotting_summary.csv"
    df.to_csv(plotting_summary_path, index=False)
    print(f"Saved plotting summary to {plotting_summary_path}")
    
    # Aggregate stats
    df_valid = df[df["status"].isin(["success", "recovered", "checkpoint_recovered"])].copy()
    
    stats_list = []
    unique_values = sorted(df["value"].unique(), key=lambda x: (isinstance(x, (int, float)), x))
    
    for val in unique_values:
        sub_all = df[df["value"] == val]
        sub_valid = df_valid[df_valid["value"] == val]
        
        n_total = len(sub_all)
        n_success = len(sub_all[sub_all["status"] == "success"])
        n_recovered = len(sub_all[sub_all["status"].isin(["recovered", "checkpoint_recovered"])])
        
        success_rate = (n_success + n_recovered) / n_total if n_total > 0 else 0
        mean_rmse = sub_valid["test_rmse"].mean() if not sub_valid.empty else float('nan')
        std_rmse = sub_valid["test_rmse"].std() if not sub_valid.empty else float('nan')
        avg_time = sub_all["train_time"].mean() if not sub_all.empty else float('nan')
        
        stats_list.append({
            "value": str(val),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "success_rate": success_rate,
            "avg_time": avg_time,
            "n_total": n_total,
            "n_valid": len(sub_valid)
        })
        
    stats_df = pd.DataFrame(stats_list)
    
    # --- BAR CHARTS (keeping existing logic) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_indices = np.arange(len(stats_df))
    axes[0].bar(x_indices, stats_df["mean_rmse"], yerr=stats_df["std_rmse"], capsize=5, color="#74A6D4", edgecolor='black')
    # axes[0].set_title(f"Mean Test RMSE (±std)")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(x_indices); axes[0].set_xticklabels(stats_df["value"], rotation=45)
    
    axes[1].bar(x_indices, stats_df["success_rate"] * 100, color="#3C6997", edgecolor='black')
    # axes[1].set_title("Success Rate (%)")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(x_indices); axes[1].set_xticklabels(stats_df["value"], rotation=45)
    
    axes[2].bar(x_indices, stats_df["avg_time"], color="#0D3B66", edgecolor='black')
    # axes[2].set_title("Average Training Time")
    axes[2].set_ylabel("Average Training Time (s)")
    axes[2].set_xticks(x_indices); axes[2].set_xticklabels(stats_df["value"], rotation=45)


    
    # fig.suptitle(f"{parameter_name} {folder_path.name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(folder_path / f"metrics_summary.png"); plt.close()
    
    # --- TRAJECTORY PLOTS ---
    for val in unique_values:
        sub_val = df_valid[df_valid["value"] == val]
        if sub_val.empty: continue
            
        run_entry = sub_val.sort_values("test_rmse").iloc[0]
        config_id = run_entry["config_id"]
        run_id = run_entry["run_id"]
        run_dir = folder_path / f"config_{config_id}" / f"run_{run_id}"
        
        # Load cached predictions
        pred_file = None
        for pf in ["test_preds_final.npy", "test_preds_checkpoint.npy", "test_preds_recovered.npy"]:
            if (run_dir / pf).exists():
                pred_file = run_dir / pf
                break
        
        if not pred_file: # Should not happen given df_valid filter, but safe fallback
            continue 
            
        try:
            test_preds = jnp.load(pred_file)
            meta_path = run_dir / "run_meta.json"
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            test_indices = meta.get("test_indices", [])[:10]
            if not test_indices: continue
            
            n_to_plot = len(test_indices)
            cols = 5
            rows = (n_to_plot + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
            axes = axes.flatten()
            
            # Map test_indices to local indices in test_preds
            all_test_indices = meta.get("test_indices", [])
            test_idx_mapping = {idx: i for i, idx in enumerate(all_test_indices)}
            
            for i, strain_idx in enumerate(test_indices):
                local_idx = test_idx_mapping[strain_idx]
                y_true_full = ys_total[strain_idx]
                y_pred = test_preds[local_idx]
                
                # Plot GT points
                axes[i].scatter(ts, y_true_full[:, pCA_index], marker='x', color='red', label='GT', alpha=0.3)
                
                # Handle timepoint mismatch for plotting predictions
                ts_pred = ts
                if y_pred.shape[0] != ts.shape[0]:
                    sub_indices = jnp.linspace(0, len(ts) - 1, y_pred.shape[0]).astype(int)
                    ts_pred = ts[sub_indices]
                    # Also highlight the GT points matching the predictions
                    axes[i].scatter(ts_pred, y_true_full[sub_indices, pCA_index], marker='o', facecolors='none', edgecolors='red', s=50)
                
                # Plot Prediction
                axes[i].plot(ts_pred, y_pred[:, pCA_index], color='blue', label='Pred', marker='.')
                
                axes[i].set_title(f"Strain {strain_idx}")
                if i == 0: axes[i].legend()
            
            # Turn off unused subplots
            for i in range(len(test_indices), len(axes)):
                axes[i].axis('off')
            
            source_label = "Final" if "final" in pred_file.name else ("Checkpoint" if "checkpoint" in pred_file.name else "Recovered")
            fig.suptitle(f"pCA Trajectories ({source_label} Model): {parameter_name} = {val} (Config {config_id}, Run {run_id})", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(folder_path / f"trajectories_config_{config_id}.png"); plt.close()
            
        except Exception as e:
            print(f"  Failed plotting trajectories for config {config_id}: {e}")
            import traceback; traceback.print_exc()

    return stats_df

    return stats_df

def main():
    MODEL_PATH = 'models/pCA_model_changed_S.xml'
    DATA_DIR = 'data'
    # EXPERIMENTS_DIR = Path('Experiments/Parameter_tuning/batch_size_over_strain_size')
    EXPERIMENTS_DIR = Path('Experiments/Parameter_tuning')
    
    if not EXPERIMENTS_DIR.exists():
        print(f"Experiments directory not found: {EXPERIMENTS_DIR}")
        return

    print(f"Loading SBML model from: {MODEL_PATH}")
    sbml_model = SBMLModel(MODEL_PATH)
    pCA_index = sbml_model.species_names.index('pcoumaric_acid') if 'pcoumaric_acid' in sbml_model.species_names else 0
    print(f"pCA index identified: {pCA_index}")
    
    print(f"Loading data from: {DATA_DIR}")
    ts, ys_total, params_total, _ = load_data_from_csvs(
        csv_dir=DATA_DIR,
        metabolites=sbml_model.species_names,
        model_name="pCA_model",
    )
    
    # List of known non-parameter folders to skip
    SKIP_FOLDERS = ["tuning_results_complete_no_static", ".ipynb_checkpoints", "__pycache__", "batch_size_over_strain_size"]
    
    for folder in sorted(list(EXPERIMENTS_DIR.iterdir())):
        if folder.is_dir() and folder.name not in SKIP_FOLDERS:
            # Check if it actually contains config_ subfolders
            if any(folder.glob("config_*")):
                # analyze_parameter_folder(folder, sbml_model, ts, ys_total, params_total, pCA_index, "batch_size")
                analyze_parameter_folder(folder, sbml_model, ts, ys_total, params_total, pCA_index, folder.name)
            else:
                print(f"Skipping folder (no config_* subfolders): {folder.name}")

if __name__ == "__main__":
    main()
