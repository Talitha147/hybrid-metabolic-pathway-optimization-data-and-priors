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

    model_name = "pCA_model"

    tracker_path = "Experiments/Question_3/experiment_tracker.csv"
    
    n_test_strains = 100

    print("Loading model and data...")
    sbmlModel = SBMLModel('models/pCA_model_changed_S.xml')
   
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
    
    strain_nums = [8, 24, 50, 100, 200, 300]
    original_metabolites = sbmlModel.species_names
    
    def run(model_path, name, M = None):

        sbmlModel_lumped = SBMLModel(model_path)
        S = sbmlModel_lumped._get_stoichiometric_matrix() 
        
        if M is None:
            M = jnp.ones(S.shape[1])
        
        model_metabolites = sbmlModel_lumped.species_names
        metabolite_indices = [original_metabolites.index(met) for met in model_metabolites]
        ys_subset = ys_total[:, :, metabolite_indices]
        data_tuple_subset = (ts, ys_subset, params_total)
    
        for ts_indices in indices_list:
            for strain_num in strain_nums:
                base_path="Experiments/Question_3/"
                experiment_name = f"pCA_{name}_{len(ts_indices)-1}_points_{strain_num}_strains"
                summary_path = Path(base_path) / experiment_name / "analysis" / "all_runs_summary.csv"
                
                if summary_path.exists():
                    print(f"Skipping {experiment_name} - already exists.")
                    continue
                    
                print(f"Running {experiment_name}...")
                runner = ExperimentRunner(model_path, ts_indices, data_tuple_subset, mask=M, metabolite_indices=metabolite_indices)
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

    model_dir = "models/pCA_model/"
    model_paths = [model_dir + "pCA_model_lumped_1_part_1.xml", model_dir + "pCA_model_lumped_2_part_1.xml", model_dir + "pCA_model_lumped_3_part_1.xml"]
    model_names = ["lumped_1_part_1", "lumped_2_part_1", "lumped_3_part_1"]
    
    for model_path, name in zip(model_paths, model_names):
        run(model_path, name)

    model_paths = [model_dir + "pCA_model_lumped_1_part_2.xml", model_dir + "pCA_model_lumped_2_part_2.xml", model_dir + "pCA_model_lumped_3_part_2.xml"]
    model_names = ["lumped_1_part_2", "lumped_2_part_2", "lumped_3_part_2"]

    # Lumped reactions are masked, others are considered known
    masks = [jnp.array([1]), jnp.array([0, 0, 0, 0, 1]), jnp.array([0, 0, 0, 0, 0, 1])]
    
    for model_path, name, mask in zip(model_paths, model_names, masks):
        run(model_path, name, mask)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
