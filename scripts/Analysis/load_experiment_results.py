import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm

def parse_experiment_name(exp_name):
    """
    Parses experiment name to extract ModelType, Steps/Points, and Strains.
    Example: pCA_XGBoost_14_steps_200_strains -> XGBoost, 14, 200
    """
    parts = exp_name.split('_')
    
    # Extract ModelType
    model_type = "Unknown"
    if "NODE" in parts or "pCANODE" in exp_name:
        model_type = "NODE"
    elif "hybrid" in parts or "Hybrid" in parts or "pCAhybrid" in exp_name:
        model_type = "Hybrid"
    elif "XGBoost" in parts or "xgboost" in parts or "pCAXGBoost" in exp_name:
        model_type = "XGBoost"
    
    # Extract Strains
    n_strains = 0
    if "strains" in parts:
        idx = parts.index("strains")
        if idx > 0 and parts[idx-1].isdigit():
            n_strains = int(parts[idx-1])
            
    # Extract Steps/Points
    steps = 0
    if "steps" in parts:
        idx = parts.index("steps")
        if idx > 0 and parts[idx-1].isdigit():
            steps = int(parts[idx-1])
    elif "points" in parts:
        idx = parts.index("points")
        if idx > 0 and parts[idx-1].isdigit():
            steps = int(parts[idx-1])
            
    return model_type, steps, n_strains

def load_all_results(base_dir="Experiments/Question_1", output_pickle="experiment_results_df.pkl", output_csv="experiment_results_summary.csv"):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist.")
        return None

    results_list = []
    
    # Recursively find all results.json files
    print(f"Searching for results.json in {base_path}...")
    result_files = list(base_path.glob("**/results.json"))
    print(f"Found {len(result_files)} files.")

    for file_path in tqdm(result_files, desc="Loading results"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract metadata from path
            # Expected path: .../ModelType/ExperimentName/subset_i/seed_j/results.json
            parts = file_path.parts
    
            seed = parts[-2]
            subset = parts[-3]
            exp_name = parts[-4]
          
            model_type_parsed, steps, n_strains = parse_experiment_name(exp_name)
           
            entry = {
                "Experiment": exp_name,
                "ModelType": model_type_parsed,
                "Steps": steps,
                "Strains": n_strains,
                "Subset": subset,
                "Seed": seed,
                "Path": str(file_path),
            }
            
            meta_path = file_path.parent / "run_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    entry["Status"] = meta.get("status")
                    entry["TrainTime"] = meta.get("train_time_seconds")
                    entry["StepsCompleted"] = meta.get("actual_steps_completed")
                    
                    # Capture evaluation exclusion info if present
                    if "eval_excluded_fraction" in meta:
                        entry["eval_excluded_fraction"] = meta["eval_excluded_fraction"]
                    if "eval_excluded_points" in meta:
                        entry["eval_excluded_points"] = meta["eval_excluded_points"]
                    if "eval_total_points" in meta:
                        entry["eval_total_points"] = meta["eval_total_points"]
                except Exception as e:
                    print(f"Warning: Could not read {meta_path}: {e}")

            for key, value in data.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    if key not in entry:
                        entry[key] = value
                elif key == "ExclusionInfo" and isinstance(value, dict):
                    # Fallback: extract from results.json if not in meta
                    entry["eval_excluded_fraction"] = value.get("eval_excluded_fraction")
                    entry["eval_excluded_points"] = value.get("eval_excluded_points")
                    entry["eval_total_points"] = value.get("eval_total_points")
                elif key in ["final_pCA_preds", "Indices", "TestIndices", "TrainIndices", "rmse_per_strain", "Train_rmse_per_strain", "ts_indices", "NRMSE_per_species", "Train_NRMSE_per_species"]:
                  
                    entry[key] = value
                elif key in ["Predictions", "GroundTruths", "Train_Predictions", "Train_GroundTruths"]:
                    entry[key] = value
                    
            pred_path = file_path.parent / "predictions.npz"
            if pred_path.exists():
                with np.load(pred_path) as npz:
                    for k in npz.files:
                        entry[k] = npz[k] 
                        
            gt_path = file_path.parent.parent / "ground_truths.npz"
            if gt_path.exists():
                with np.load(gt_path) as npz:
                    for k in npz.files:
                        entry[k] = npz[k] 
            
            results_list.append(entry)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print("\nCreating DataFrame...")
    df = pd.DataFrame(results_list)
    
    print(f"Saving full Pickle to {base_dir}/{output_pickle}...")
    df.to_pickle(f"{base_dir}/{output_pickle}")
    print(f"Saved full DataFrame to {base_dir}/{output_pickle}")
    
    print("Flattening metrics for summary CSV...")
    metabolite_names = ["substrate", "c_biomass", "co2", "e4p", "pep", "dhap", "epsp", "pcoumaric_acid"]
    for col in ["NRMSE_per_species", "Train_NRMSE_per_species"]:
        if col in df.columns:
            for i, name in enumerate(metabolite_names):
                df[f"{col}_{name}"] = df[col].apply(lambda x: x[i] if isinstance(x, list) and len(x) > i else np.nan)

    summary_cols = [c for c in df.columns if not isinstance(df[c].iloc[0], (list, dict, np.ndarray))]
    
    if "Path" not in summary_cols and "Path" in df.columns:
        summary_cols.append("Path")
        
    print(f"Saving summary CSV to {base_dir}/{output_csv}...")
    df[summary_cols].to_csv(f"{base_dir}/{output_csv}", index=False)
    print(f"Saved summary CSV ({len(summary_cols)} columns) to {base_dir}/{output_csv}")
    
    return df

if __name__ == "__main__":
    load_all_results()
