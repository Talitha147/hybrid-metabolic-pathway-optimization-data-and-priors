import os


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.numpy as jnp
from pathlib import Path
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from models.NODE import NODEModel
from data_generation import load_data_from_csvs
import jax
from ExperimentRunner import ExperimentRunner
import gc
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run experiments for Question 1")
    parser.add_argument("--model_type", type=str, choices=["Hybrid", "NODE", "XGBoost", "All"], default="All", help="Model type to run")
    parser.add_argument("--points_idx", type=int, default=-1, help="Index of the ts_indices list to use (-1 for all)")
    parser.add_argument("--strain_idx", type=int, default=-1, help="Index of the strain_nums list to use (-1 for all)")
    args = parser.parse_args()

    model_path = 'models/pCA_model_changed_S.xml'
    model_name = "pCA_model"
    
    n_test_strains = 100
    
    tracker_path = "Experiments/Question_1/experiment_tracker.csv"

    print("Loading model and data...")
    sbmlModel = SBMLModel(model_path)
    kinetic_model = jax.jit(sbmlModel.get_kinetic_model())
    
    S = sbmlModel._get_stoichiometric_matrix() 
    
    # Load the data
    ts, ys_total, params_total, _ = load_data_from_csvs(
        csv_dir="data",
        metabolites= sbmlModel.species_names,
        model_name = model_name,
    )
   
    
    ts = jnp.arange(40)
    
    indices_30 = jnp.linspace(0, len(ts) - 1, 31).astype(int)
    indices_15 = jnp.linspace(0, len(ts) - 1, 15).astype(int)
    indices_7 = jnp.linspace(0, len(ts) - 1, 8).astype(int)
    indices_3 = jnp.linspace(0, len(ts) - 1, 4).astype(int)
    
    indices_list = [jnp.array([0, -1]), indices_3, indices_7, indices_15, indices_30]
    
    M = jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    
    hyper_params = {
        "width_NODE": 192,
        "depth_NODE": 1,
        "normalize": True,
        "dropout": 0.2,
        "batch_size": 256,
        "learning_rate": 5e-5,
        "solver": "Tsit5",
        "weight_decay": 1e-6,
        "dt0": 1e-9,
        "rtol": 1e-4,
        "atol": 1e-6
    }
    
    strain_nums = [8, 24, 50, 200, 300, 500]
    # strain_nums = [8, 24, 200]
    data_tuple = (ts, ys_total, params_total)
    
    xgb_hyper_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.6,
        "colsample_bytree": 1.0
    }

    # Determine which points and strains to run
    if args.points_idx != -1:
        indices_list = [indices_list[args.points_idx]]
    
    if args.strain_idx != -1:
        strain_nums = [strain_nums[args.strain_idx]]

    # 1. Hybrid Model
    for ts_indices in indices_list:
        if args.model_type not in ["Hybrid", "All"]: break
        for strain_num in strain_nums:
            base_path="Experiments/Question_1/Hybrid"
            experiment_name = f"pCA_hybrid_{len(ts_indices)-1}_points_{strain_num}_strains"
            summary_path = Path(base_path) / experiment_name / "analysis" / "all_runs_summary.csv"
            
            if summary_path.exists():
                print(f"Skipping {experiment_name} - already exists.")
                continue
                
            print(f"Running {experiment_name}...")
            runner = ExperimentRunner(model_path, ts_indices, data_tuple, mask=M)
            results_df = runner.train_models_on_random_subsets(
                hyper_parameters= hyper_params,
                n_subsets=10,
                n_seeds_per_subset=5,
                n_train_strains=strain_num + 50,
                n_test_strains=n_test_strains, 
                experiment_name=experiment_name,
                base_path=base_path,
                max_steps=5000,
                patience=500,
                val_split=0.0,
                n_val_strains=50,
                checkpoint_freq=200,
                timeout_seconds=7200,
                make_plots=False,
                tracker_path=tracker_path,
            )
          
            del runner
            gc.collect()
            jax.clear_caches()
            
    # 2. NODE Model
    for ts_indices in indices_list:   
        if args.model_type not in ["NODE", "All"]: break
        for strain_num in strain_nums:
            base_path="Experiments/Question_1/NODE"
            experiment_name = f"pCA_NODE_{len(ts_indices)-1}_points_{strain_num}_strains"
            summary_path = Path(base_path) / experiment_name / "analysis" / "all_runs_summary.csv"
            
            if summary_path.exists():
                print(f"Skipping {experiment_name} - already exists.")
                continue
                
            print(f"Running {experiment_name}...")
            runner = ExperimentRunner(model_path, ts_indices, data_tuple, model_class=NODEModel)
            results_df = runner.train_models_on_random_subsets(
                hyper_parameters= hyper_params,
                n_subsets=10,
                n_seeds_per_subset=5,
                n_train_strains=strain_num + 50,
                n_test_strains=n_test_strains, 
                experiment_name=experiment_name,
                base_path=base_path,
                max_steps=5000,
                patience=500,
                val_split=0.0,
                n_val_strains=50,
                checkpoint_freq=200,
                timeout_seconds=7200,
                make_plots=False,
                tracker_path=tracker_path,
            )
    
            del runner
            gc.collect()
            jax.clear_caches()
            
    # 3. XGBoost
    for ts_indices in indices_list:
        if args.model_type not in ["XGBoost", "All"]: break
        for strain_num in strain_nums:
            experiment_name = f"pCA_XGBoost_{len(ts_indices)-1}_steps_{strain_num}_strains"
            summary_path = Path("Experiments/Question_1/XGBoost") / experiment_name / "analysis" / "all_runs_summary.csv"
            
            if summary_path.exists() or (Path("Experiments/Question_1/XGBoost") / experiment_name).exists():
                print(f"Skipping {experiment_name} - already exists.")
                continue
       
            print(f"Running {experiment_name}...")
            runner = ExperimentRunner(
                model_path=model_path,
                ts_indices=ts_indices,    
                data_tuple=data_tuple,
                model_class=ExperimentRunner.XGBOOST,   
            )
            
            results_df = runner.train_models_on_random_subsets(
                        n_subsets=10,
                        n_seeds_per_subset=5,
                        hyper_parameters= xgb_hyper_params,
                        n_train_strains=strain_num + 50,
                        n_test_strains=n_test_strains, 
                        base_path="Experiments/Question_1/XGBoost",
                        experiment_name=experiment_name,
                        max_steps=5000,
                        patience=500,
                        val_split=0.0,
                        n_val_strains=50,
                        checkpoint_freq=200,
                        timeout_seconds=7200,
                        make_plots=False,
                    )
            
            
            del runner
            gc.collect()
            jax.clear_caches()
            
    print("\nAll specified experiments completed.")

if __name__ == "__main__":
    main()
