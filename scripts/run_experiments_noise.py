import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.numpy as jnp
from pathlib import Path
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from data_generation import load_data_from_csvs
import jax
from ExperimentRunner import ExperimentRunner
import gc
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run noise experiments with multiple masking levels (Hybrid Model)")
    parser.add_argument("--noise", type=int, choices=[5, 10, 20], help="Specific noise level to run (if not provided, runs all 3)")
    parser.add_argument("--points_idx", type=int, default=-1, help="Index of the ts_indices list to use (-1 for all)")
    parser.add_argument("--strain_idx", type=int, default=-1, help="Index of the strain_nums list to use (-1 for all)")
    args = parser.parse_args()

    model_path = 'models/pCA_model_changed_S.xml'
    model_name = "pCA_model"
    
    n_test_strains = 100

    print("Loading model...")
    sbmlModel = SBMLModel(model_path)
    
    ts = jnp.arange(40)
    
    indices_30 = jnp.linspace(0, len(ts) - 1, 31).astype(int)
    indices_15 = jnp.linspace(0, len(ts) - 1, 15).astype(int)
    indices_7 = jnp.linspace(0, len(ts) - 1, 8).astype(int)
    indices_3 = jnp.linspace(0, len(ts) - 1, 4).astype(int)
    
    indices_list = [jnp.array([0, -1]), indices_3, indices_7, indices_15, indices_30]
    
    # Masks (representing different levels of structural knowledge)
    M = jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    M3 = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    M5 = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])    

    masks = [M, M3, M5]
    mask_names = ["Original", "Reactions_to_dahp_known", "Only_product_unknown"]
    
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
    
    strain_nums = [8, 24, 50, 100, 200, 300]
    
    if args.points_idx != -1:
        indices_list = [indices_list[args.points_idx]]
    
    if args.strain_idx != -1:
        strain_nums = [strain_nums[args.strain_idx]]

    noise_levels = [5, 10, 20]
    if args.noise:
        noise_levels = [args.noise]

    # Load the true data for evaluation
    print(f"Loading true data from 'data' for evaluation...")
    ts_true, ys_true, params_true, _ = load_data_from_csvs(
        csv_dir="data",
        metabolites=sbmlModel.species_names,
        model_name=model_name,
    )
    true_data_tuple = (ts_true, ys_true, params_true)

    for noise_level in noise_levels:
        print(f"\n========================================")
        print(f"Starting experiments for {noise_level}% noise")
        print(f"========================================")
        
        data_dir = f"data_noise_{noise_level}"
        
        # Load the noisy data
        print(f"Loading noisy data from {data_dir}...")
        ts_data, ys_total, params_total, _ = load_data_from_csvs(
            csv_dir=data_dir,
            metabolites=sbmlModel.species_names,
            model_name=model_name,
        )
        data_tuple = (ts_data, ys_total, params_total)

        for mask, mask_name in zip(masks, mask_names):
            print(f"\n--- Masking Level: {mask_name} ---")
            
            base_path = f"Experiments/Noise_{noise_level}/{mask_name}"
            tracker_path = f"Experiments/Noise_{noise_level}/{mask_name}/experiment_tracker.csv"
            
            os.makedirs(base_path, exist_ok=True)
            
            for ts_indices in indices_list:
                for strain_num in strain_nums:
                    experiment_name = f"pCA_{mask_name}_{len(ts_indices)-1}_points_{strain_num}_strains"
                    summary_path = Path(base_path) / experiment_name / "analysis" / "all_runs_summary.csv"
                    
                    if summary_path.exists():
                        print(f"Skipping {experiment_name} ({noise_level}% noise) - already exists.")
                        continue
                        
                    print(f"Running {experiment_name} with {noise_level}% noise and {mask_name} mask...")
                    runner = ExperimentRunner(model_path, ts_indices, data_tuple, mask=mask, true_data_tuple=true_data_tuple)
                    results_df = runner.train_models_on_random_subsets(
                        hyper_parameters=hyper_params,
                        n_subsets=10,
                        n_seeds_per_subset=3,
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

    print("\nAll specified noise experiments completed.")

if __name__ == "__main__":
    main()
