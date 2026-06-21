import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from sklearn.model_selection import RandomizedSearchCV, KFold

# Add project root to path
script_path = Path(__file__).parent.absolute()
project_root = script_path.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from scripts.models.XGBoostModel import XGBoost
from scripts.data_generation import load_data_from_csvs
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel

def prepare_xgboost_data(ts, ys, params, ts_indices):

    n_strains = ys.shape[0]
    n_obs = len(ts_indices)
    n_species = ys.shape[2]
    
    y0_all = ys[:, 0, :] 
    ys_obs = ys[:, ts_indices, :] 
    
   
    params_expanded = np.repeat(params, n_obs, axis=0)
    y0_expanded = np.repeat(y0_all, n_obs, axis=0)
    
    obs_times = ts[ts_indices].reshape(-1, 1)
    times_expanded = np.tile(obs_times, (n_strains, 1))
    
    X_full = np.concatenate([params_expanded, times_expanded, y0_expanded], axis=1)
    
    Y = ys_obs.reshape(-1, n_species)
    
    return X_full, Y

def main():
    MODEL_PATH = 'models/pCA_model_changed_S.xml'
    DATA_DIR = 'data'
    OUTPUT_DIR = Path("Experiments/Parameter_tuning/xgboost_results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load metadata and data
    print(f"Loading SBML model from: {MODEL_PATH}")
    sbml_model = SBMLModel(MODEL_PATH)
    
    print(f"Loading data from: {DATA_DIR}")
    ts, ys_total, params_total, _ = load_data_from_csvs(
        csv_dir=DATA_DIR,
        metabolites=sbml_model.species_names,
        model_name="pCA_model",
    )
    
    # Use the same 7-point/400-strain subset used for Hybrid tuning
    indices_7 = np.linspace(0, len(ts) - 1, 8).astype(int)
    n_samples = min(400, ys_total.shape[0])
    
    print(f"Preparing data for expansion (400 strains, 7 steps)...")
    X, Y = prepare_xgboost_data(
        ts, 
        ys_total[:n_samples], 
        params_total[:n_samples], 
        indices_7
    )
    
    print(f"XGBoost Tuning Data Shape: X={X.shape}, Y={Y.shape}")
    
    # Define Parameter Grid
    param_dist = {
        'n_estimators': [100, 200, 500, 800, 1000],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1],
        'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1]
    }
    
    
    model = XGBoost()
    
    print("Starting Randomized Search (50 iterations, 5-fold CV)...")
    start_time = time.time()
    
    search = RandomizedSearchCV(
        model, 
        param_distributions=param_dist, 
        n_iter=1000, 
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1, 
        random_state=42
    )
    
   
    search.fit(X, Y)
    
    tuning_time = time.time() - start_time
    print(f"\nTuning completed in {tuning_time/60:.2f} minutes")
    print(f"Best Score (RMSE): {-search.best_score_:.6f}")
    print(f"Best Params: {search.best_params_}")
    
    cv_results = search.cv_results_
    tuning_data = []
    
    for i in range(len(cv_results['params'])):
        entry = {
            "config_id": i,
            "status": "success",
            "test_rmse": -cv_results['mean_test_score'][i], 
            "training_time": cv_results['mean_fit_time'][i],
            "std_test_score": cv_results['std_test_score'][i],
            "rank": cv_results['rank_test_score'][i],
        }
        # Add the hyperparameters as flat columns
        for param_name, param_value in cv_results['params'][i].items():
            entry[param_name] = param_value
            
        tuning_data.append(entry)
    
    results_df = pd.DataFrame(tuning_data)
    results_df.sort_values("test_rmse", inplace=True)
    
    # Save standard tuning_results.csv
    results_df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)
    
    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=2)
        
    print(f"\nResults saved to {OUTPUT_DIR}")
    print("You can now update your XGBoost experiments with these parameters.")

if __name__ == "__main__":
    main()
