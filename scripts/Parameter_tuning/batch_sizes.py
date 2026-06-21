import sys
import os
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
import time
import jax

script_path = Path(__file__).parent.absolute()
project_root = script_path.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from scripts.models.hybrid_model import HybridModel
from hyperparameter_tuning import HyperparameterTuner
from scripts.data_generation import load_data_from_csvs

   
    

def main():
    MODEL_PATH = 'models/pCA_model_changed_S.xml'
    DATA_DIR = 'data'
  
    print(f"Loading SBML model from: {MODEL_PATH}")
    sbml_model = SBMLModel(MODEL_PATH)
    
    # Load the data
    print(f"Loading data from: {DATA_DIR}")
    ts, ys_total, params_total, _ = load_data_from_csvs(
        csv_dir=DATA_DIR,
        metabolites=sbml_model.species_names,
        model_name="pCA_model",
    )
    
    

    mask = jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]) 

    base_params = {
        "filepath": MODEL_PATH,
        "mask": mask,
        "seed": 42
    }

    def run_tuning_individual_parameter(parameter_name: str, batch_sizes, train_size, total_size):

      
        param_grid = {
            "width_NODE": [73],
            "depth_NODE": [1],
            "normalize": [True],
            "dropout": [0.1],
            "batch_size": batch_sizes,
            "learning_rate": [1e-5],
            "solver": ["Tsit5"],
            "weight_decay": [0.0],
            "dt0": [1e-4],
            "rtol": [1.5e-4],
            "atol": [2e-6]
        }

        # Select 400-strain subset for tuning (250 train + 50 val + 100 test)
        indices_7 = jnp.linspace(0, len(ts) - 1, 8).astype(int)
        n_samples = min(total_size, ys_total.shape[0])
        
        ts_train = ts[indices_7]
        ys_train = ys_total[:n_samples, indices_7, :]
        params_train = params_total[:n_samples]

        print(f"Training data shape: {ys_train.shape}")
        
        N_ITER = 20
        N_RUNS = 8
        MAX_STEPS = 2000
        PATIENCE = 300
        USE_HILL_CLIMBING = False
        HILL_CLIMBING_STEPS = 5

        OUTPUT_DIR = f"Experiments/Parameter_tuning/batch_size_over_strain_size/{parameter_name}"
        print(f"Starting HyperparameterTuner with n_iter={N_ITER}, n_runs={N_RUNS}")
        tuner = HyperparameterTuner(HybridModel, base_params, param_grid)

       

        results = tuner.tune(
            ts_train, 
            ys_train, 
            params_train, 
            max_steps=MAX_STEPS, 
            patience=PATIENCE, 
            output_dir=OUTPUT_DIR,
            use_hill_climbing=USE_HILL_CLIMBING,
            hill_climbing_steps=HILL_CLIMBING_STEPS,
            n_iter=N_ITER,
            timeout=7200,
            n_runs=N_RUNS,
            n_train_pure=train_size,
            n_val_pure=50
        )

        # Display results sorted by Test RMSE
        print(f"\nTop 10 Configurations for {parameter_name} (Sorted by Test RMSE):")
        if not results.empty:
            sorted_results = results.sort_values("test_rmse")
            print(sorted_results.head(10))
            
            # Save summary to CSV
            try:
                results_csv = Path(OUTPUT_DIR) / "tuning_summary.csv"
                sorted_results.to_csv(results_csv, index=False)
                print(f"\nSummary saved to {results_csv}")
            except Exception as e:
                print(f"Workers collided writing summary, but that's okay: {e}")

        else:
            print("No successful runs completed.")

    run_tuning_individual_parameter("300_strains", [16, 32, 64, 128, 192, 256, 300], train_size=300, total_size=450)
    run_tuning_individual_parameter("500_strains", [16, 32, 64, 128, 192, 256, 320, 500], train_size=500, total_size=650)
    run_tuning_individual_parameter("200_strains", [8, 16, 32, 64, 128, 200], train_size=200, total_size=350)
    run_tuning_individual_parameter("150_strains", [8, 16, 32, 64, 128, 150], train_size=150, total_size=300)
    run_tuning_individual_parameter("100_strains", [4, 8, 16, 32, 64, 100], train_size=100, total_size=250)
    run_tuning_individual_parameter("50_strains", [2, 4, 8, 16, 32, 50], train_size=50, total_size=200)
    run_tuning_individual_parameter("24_strains", [1, 2, 4, 8, 16, 24], train_size=24, total_size=399)
    run_tuning_individual_parameter("12_strains", [1, 2, 4, 8, 12], train_size=12, total_size=348)
    run_tuning_individual_parameter("8_strains", [1, 2, 4, 8], train_size=8, total_size=297)
    

   
if __name__ == "__main__":
    main()
