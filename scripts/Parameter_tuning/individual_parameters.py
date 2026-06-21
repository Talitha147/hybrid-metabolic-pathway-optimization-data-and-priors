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
from scripts.Parameter_tuning.hyperparameter_tuning import HyperparameterTuner
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
    
    # Select 400-strain subset for tuning (250 train + 50 val + 100 test)
    indices_7 = jnp.linspace(0, len(ts) - 1, 8).astype(int)
    n_samples = min(400, ys_total.shape[0])
    
    ts_train = ts[indices_7]
    ys_train = ys_total[:n_samples, indices_7, :]
    params_train = params_total[:n_samples]

    print(f"Training data shape: {ys_train.shape}")

    mask = jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]) 

    base_params = {
        "filepath": MODEL_PATH,
        "mask": mask,
        "seed": 42
    }

    def run_tuning_individual_parameter(parameter_name: str, param_grid):

        N_ITER = 20
        N_RUNS = 8
        MAX_STEPS = 2000
        PATIENCE = 300
        USE_HILL_CLIMBING = False
        HILL_CLIMBING_STEPS = 5

        OUTPUT_DIR = f"Experiments/Parameter_tuning/{parameter_name}"
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
            n_train_pure=250,
            n_val_pure=50
        )

        # Display results sorted by Test RMSE
        print("\nTop 10 Configurations (Sorted by Test RMSE):")
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


    param_grid_batch_size = {
        "width_NODE": [73],
        "depth_NODE": [1],
        "normalize": [True],
        "dropout": [0.1],
        "batch_size": [8, 16, 32, 64, 128, 256],
        "learning_rate": [1e-5],
        "solver": ["Tsit5"],
        "weight_decay": [0.0],
        "dt0": [1e-4],
        "rtol": [1.5e-4],
        "atol": [2e-6]
    }

    run_tuning_individual_parameter("batch_size", param_grid_batch_size)

    param_grid_learning_rate = {
        "width_NODE": [73],
        "depth_NODE": [1],
        "normalize": [True],
        "dropout": [0.1],
        "batch_size": [128],
        "learning_rate": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
        "solver": ["Tsit5"],
        "weight_decay": [0.0],
        "dt0": [1e-4],
        "rtol": [1.5e-4],
        "atol": [2e-6]
    }

    run_tuning_individual_parameter("learning_rate", param_grid_learning_rate)

    param_grid_solver = {
        "width_NODE": [73],
        "depth_NODE": [1],
        "normalize": [True],
        "dropout": [0.1],
        "batch_size": [128],
        "learning_rate": [1e-5],
        "solver": ["Kvaerno3", "Kvaerno5", "Tsit5", "Dopri5", "Heun", "Euler"],
        "weight_decay": [0.0],
        "dt0": [1e-4],
        "rtol": [1.5e-4],
        "atol": [2e-6]
    }

    run_tuning_individual_parameter("solver", param_grid_solver)

    param_grid_weight_decay = {
        "width_NODE": [73],
        "depth_NODE": [1],
        "normalize": [True],
        "dropout": [0.1],
        "batch_size": [128],
        "learning_rate": [1e-5],
        "solver": ["Tsit5"],
        "weight_decay": [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        "dt0": [1e-4],
        "rtol": [1e-3],
        "atol": [1e-5]
    }

    run_tuning_individual_parameter("weight_decay", param_grid_weight_decay)

    param_grid_dt0 = {
        "width_NODE": [73],
        "depth_NODE": [1],
        "normalize": [True],
        "dropout": [0.1],
        "batch_size": [128],
        "learning_rate": [1e-5],
        "solver": ["Tsit5"],
        "weight_decay": [0.0],
        "dt0": [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11],
        "rtol": [1.5e-4],
        "atol": [2e-6]
    }

    run_tuning_individual_parameter("dt0", param_grid_dt0)

    param_grid_rtol = {
        "width_NODE": [73],
        "depth_NODE": [1],
        "normalize": [True],
        "dropout": [0.1],
        "batch_size": [128],
        "learning_rate": [1e-5],
        "solver": ["Tsit5"],
        "weight_decay": [0.0],
        "dt0": [1e-4],
        "rtol": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        "atol": [2e-6]
    }

    run_tuning_individual_parameter("rtol", param_grid_rtol)

    param_grid_atol = {
        "width_NODE": [73],
        "depth_NODE": [1],
        "normalize": [True],
        "dropout": [0.1],
        "batch_size": [128],
        "learning_rate": [1e-5],
        "solver": ["Tsit5"],
        "weight_decay": [0.0],
        "dt0": [1e-4],
        "rtol": [1.5e-4],
        "atol": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    }

    run_tuning_individual_parameter("atol", param_grid_atol)

    param_grid_width_NODE = {
        "width_NODE": [8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512],
        "depth_NODE": [1],
        "normalize": [True],
        "dropout": [0.1],
        "batch_size": [128],
        "learning_rate": [1e-5],
        "solver": ["Tsit5"],
        "weight_decay": [0.0],
        "dt0": [1e-4],
        "rtol": [1.5e-4],
        "atol": [2e-6]
    }

    run_tuning_individual_parameter("width_NODE", param_grid_width_NODE)

    param_grid_depth_NODE = {
        "width_NODE": [73],
        "depth_NODE": [1, 2, 3, 4, 5],
        "normalize": [True],
        "dropout": [0.1],
        "batch_size": [128],
        "learning_rate": [1e-5],
        "solver": ["Tsit5"],
        "weight_decay": [0.0],
        "dt0": [1e-4],
        "rtol": [1.5e-4],
        "atol": [2e-6]
    }

    run_tuning_individual_parameter("depth_NODE", param_grid_depth_NODE)

    param_grid_dropout = {
        "width_NODE": [73],
        "depth_NODE": [1],
        "normalize": [True],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
        "batch_size": [128],
        "learning_rate": [1e-5],
        "solver": ["Tsit5"],
        "weight_decay": [0.0],
        "dt0": [1e-4],
        "rtol": [1.5e-4],
        "atol": [2e-6]
    }

    run_tuning_individual_parameter("dropout", param_grid_dropout)

    param_grid_normalize = {
        "width_NODE": [73],
        "depth_NODE": [1],
        "normalize": [True, False],
        "dropout": [0.1],
        "batch_size": [128],
        "learning_rate": [1e-5],
        "solver": ["Tsit5"],
        "weight_decay": [0.0],
        "dt0": [1e-4],
        "rtol": [1.5e-4],
        "atol": [2e-6]
    }

    run_tuning_individual_parameter("normalize", param_grid_normalize)

   

 

if __name__ == "__main__":
    main()
