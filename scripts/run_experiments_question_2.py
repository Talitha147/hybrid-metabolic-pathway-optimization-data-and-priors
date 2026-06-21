import pandas as pd
import jax.numpy as jnp
from pathlib import Path
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from models.NODE import NODEModel
from data_generation import load_data_from_csvs
import equinox as eqx
import jax
import jax.random as jr
import jax.nn as jnn
import diffrax
import optax
import matplotlib.pyplot as plt
from models.hybrid_model import HybridModel
import numpy as np
from ExperimentRunner import ExperimentRunner
import gc
import sys

def main():

    model_path = 'models/pCA_model_changed_S.xml'
    model_name = "pCA_model"

    tracker_path = "Experiments/Question_2/experiment_tracker.csv"
    
    n_test_strains = 100

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
    
    strain_nums = [8, 24, 50, 100, 200, 300, 500]
    data_tuple = (ts, ys_total, params_total)
    
    def run(mask, name):
    
        for ts_indices in indices_list:
            for strain_num in strain_nums:
                base_path="Experiments/Question_2/"
                experiment_name = f"pCA_{name}_{len(ts_indices)-1}_points_{strain_num}_strains"
                summary_path = Path(base_path) / experiment_name / "analysis" / "all_runs_summary.csv"
                
                if summary_path.exists():
                    print(f"Skipping {experiment_name} - already exists.")
                    continue
                    
                print(f"Running {experiment_name}...")
                runner = ExperimentRunner(model_path, ts_indices, data_tuple, mask=mask)
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


    # Masks

    # Everything masked
    M1 = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # All reactions from substrate directly, known
    M2 = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    # Reactions to dahp known
    M3 = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Only product and sink unknown
    M4 = jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    # Only product unknown
    M5 = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])    

    masks = [M1, M2, M3, M4, M5]
    mask_names = ["All_masked", "All_known_from_substrate", "Reactions_to_dahp_known", "Only_product_and_sink_unknown", "Only_product_unknown"]
    
    for mask, name in zip(masks, mask_names):
        run(mask, name)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
