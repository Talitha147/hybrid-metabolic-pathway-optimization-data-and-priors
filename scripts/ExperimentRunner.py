
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from pathlib import Path
import os
import fcntl
from scripts.models.hybrid_model import HybridModel
from scripts.models.NODE import NODEModel
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import time
import json
import pickle
import jax.numpy as jnp
import matplotlib.pyplot as plt 
import gc
from scripts.models.XGBoostModel import XGBoost



class ExperimentRunner:
   
    XGBOOST = "XGBoost"

    def __init__(self, model_path: str, ts_indices, data_tuple, mask: list[int] = None, model_class=HybridModel, metabolite_indices=None, true_data_tuple=None):
        """
        model_path: path to SBML model xml (used for HybridModel/NODEModel; ignored for XGBoost)
        data_tuple: (ts, ys, params, ...) from load_data_from_csvs
        mask: binary mask for reactions (used for HybridModel/NODEModel)
        model_class: The model class to use for experiments. Pass ExperimentRunner.XGBOOST for XGBoost.
        metabolite_indices: List of metabolite indices to include in evaluation. If None, includes all.
        true_data_tuple: To evaluate against true data instead of observed ys, pass (ts, true_ys, true_params) as true_data_tuple. Must have same shape as data_tuple. If None, evaluates against observed ys.
        """
        self.model_path = model_path
        self.ts_indices = np.array(ts_indices)
        self.ts, self.ys, self.params = data_tuple
        if true_data_tuple is not None:
            _, self.true_ys, self.true_params = true_data_tuple
        else:
            self.true_ys, self.true_params = self.ys, self.params
        self.model_class = model_class
        self.is_xgboost = (model_class == self.XGBOOST)

        self.metabolite_indices = metabolite_indices
        
        if not self.is_xgboost:
            self.sbml_model = SBMLModel(model_path)
            self.species_names = self.sbml_model.species_names
            self.pCA_index = self.species_names.index('pcoumaric_acid')

            if mask is None:
                n_reactions = self.sbml_model._get_stoichiometric_matrix().shape[1]
                self.mask = jnp.zeros(n_reactions)
            else:
                self.mask = jnp.array(mask)
        else:
            try:
                self.sbml_model = SBMLModel(model_path)
                self.species_names = self.sbml_model.species_names
            except Exception:
                n_species = self.ys.shape[2]
                self.species_names = [f"species_{k}" for k in range(n_species)]
            self.pCA_index = self.species_names.index('pcoumaric_acid') if 'pcoumaric_acid' in self.species_names else 0
            self.mask = None

        # Metabolite ranges computed from ALL data to ensure fair comparisons across subsets
        self.metabolite_ranges = (
                    np.max(self.ys, axis=(0, 1)) - np.min(self.ys, axis=(0, 1))
                ) 
        self.metabolite_ranges = np.where(
                    self.metabolite_ranges == 0, 1e-10, self.metabolite_ranges
                )


    def get_random_subset(self, n_train_strains: int, n_test_strains: int, seed: int = 0):
        """
        Returns a random subset of strains for training and testing.
        """
        total_strains = self.ys.shape[0]
        indices = np.arange(total_strains)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        if n_train_strains < 1:
             raise ValueError(f"n_train_strains must be at least 1, got {n_train_strains}")
             
        if n_train_strains + n_test_strains > total_strains:
             raise ValueError(f"Total strains requested ({n_train_strains} train + {n_test_strains} test) > available strains ({total_strains})")
        
        train_idx = indices[:n_train_strains]
        test_idx = indices[-n_test_strains:]
        
        return train_idx, test_idx



    def _make_xgboost_model(self, hyper_parameters: dict, seed: int):
        """Instantiate an XGBoost model from hyper_parameters."""

        xgb_params = {
            "random_state": seed,
            "n_estimators": hyper_parameters.get("n_estimators", 100),
            "max_depth": hyper_parameters.get("max_depth", 3),
            "learning_rate": hyper_parameters.get("learning_rate", 0.1),
        }
        return XGBoost(**xgb_params)

    def _get_ts_target_idx(self):
        """Resolve ts_indices[-1] as a positive absolute index into self.ts."""
        last = int(np.array(self.ts_indices).tolist()[-1])
        return last if last >= 0 else int(last + len(self.ts))

    def _train_xgboost(self, model, train_idx, train_ys, train_params):
        """Train the XGBoost model and return (loss_history, train_time).
        
        Uses all metabolites in ys and corresponding parameters. Treats each index
        in self.ts_indices as a separate training sample to utilize all available data.
        """
        n_strains = train_ys.shape[0]
        n_obs = len(self.ts_indices)
        
        # All species at t=0 as initial condition features
        y0_all = train_ys[:, 0, :] # (n_strains, n_species)
        
        # Targets are all metabolites at all ts_indices
        train_ys_obs = train_ys[:, self.ts_indices, :] # (n_strains, n_obs, n_species)
        
        # Flatten time dimension to create multiple training samples per strain
        # X features: [parameters, time_val, y0_all]
        train_params_expanded = np.repeat(train_params, n_obs, axis=0) # (n_strains * n_obs, n_params)
        y0_expanded = np.repeat(y0_all, n_obs, axis=0)               # (n_strains * n_obs, n_species)
        
        # Time values as extra feature
        obs_times = self.ts[self.ts_indices].reshape(-1, 1) # (n_obs, 1)
        times_expanded = np.tile(obs_times, (n_strains, 1)) # (n_strains * n_obs, 1)
        
        # Combine parameters and time into the first input argument of model.train
        X_combined = np.concatenate([train_params_expanded, times_expanded], axis=1)
        
        # Target for each expanded sample
        Y_flat = train_ys_obs.reshape(-1, train_ys_obs.shape[-1]) # (n_strains * n_obs, n_species)
        
        start = time.time()
        model.train(X_combined, Y_flat, y0s=y0_expanded)
        train_time = time.time() - start
        
        return [], train_time

   

    def train_models_on_random_subsets(
        self,
        hyper_parameters: dict,
        n_train_strains: int,
        n_test_strains: int,
        experiment_name: str,
        n_subsets: int = None,
        n_seeds_per_subset: int = 1,
        n_models: int = None,          # backward-compat alias for n_subsets
        base_path: str = "Experiments",
        max_steps: int = 2000,
        checkpoint_freq: int = 100,
        val_split: float = 0.0,
        n_val_strains: int = 0,
        patience: int = 50,
        timeout_seconds: int = 3600,
        print_freq: int = 50,
        make_plots: bool = True,
        tracker_path: str = None,
    ):

        if n_models is not None and n_subsets is None:
            n_subsets = n_models
        elif n_subsets is None:
            raise ValueError("Must provide n_subsets")

        base_path = Path(base_path) / experiment_name
        base_path.mkdir(parents=True, exist_ok=True)

        subset_summaries = []

        try:
            for subset_idx in range(n_subsets):
                print(f"Subset {subset_idx + 1}/{n_subsets}")
          
                subset_dir = base_path / f"subset_{subset_idx}"
                subset_dir_created = False

                # Using subset_idx as seed ensures all model types see identical splits.
                train_idx, test_idx = self.get_random_subset(
                    n_train_strains, n_test_strains, seed=subset_idx
                )

                if n_val_strains > 0:
                    n_val_internal = n_val_strains
                    # Use the first n_val_strains to ensure consistent validation sets across different n_train_strains
                    val_idx_internal = train_idx[:n_val_internal]
                    train_idx_actual = train_idx[n_val_internal:]
                elif val_split > 0.0:
                    val_seed_rng = np.random.default_rng(subset_idx + 10000)  
                    shuffled_train_idx = train_idx.copy()
                    val_seed_rng.shuffle(shuffled_train_idx)
                    n_val_internal = int(len(train_idx) * val_split)
                    val_idx_internal = shuffled_train_idx[-n_val_internal:]
                    train_idx_actual = shuffled_train_idx[:-n_val_internal]
                else:
                    val_idx_internal = np.array([], dtype=int)
                    train_idx_actual = train_idx

                # Pre-slice data so that we can pass training and validation sets explicitly.
                # This ensures the model uses the exact same strains recorded in subset_meta.json.
                train_ys_actual = self.ys[train_idx_actual]
                train_params_actual = self.params[train_idx_actual]
                val_ys_internal_data = self.ys[val_idx_internal]
                val_params_internal_data = self.params[val_idx_internal]
                
                test_ys = self.ys[test_idx]
                test_params = self.params[test_idx]
                

                if not self.is_xgboost and (jnp.any(jnp.isnan(train_ys_actual)) or jnp.any(jnp.isnan(val_ys_internal_data))):
                    print("WARNING: Training/Validation data contains NaNs.")

               
                subset_meta = {
                    "subset_idx": subset_idx,
                    "train_indices": train_idx.tolist(),
                    "train_indices_actual": train_idx_actual.tolist(),
                    "validation_indices": val_idx_internal.tolist(),
                    "test_indices": test_idx.tolist(),
                    "n_train_total": n_train_strains,
                    "n_train_actual": len(train_idx_actual),
                    "n_validation": len(val_idx_internal),
                    "n_test": n_test_strains,
                }
                # Deferred writing of subset_meta.json until a seed starts

                
                seed_results = []
                losses_subset = []
                val_losses_subset = []

                for seed_idx in range(n_seeds_per_subset):
                    print(f"\n  Subset {subset_idx+1}/{n_subsets} | Seed {seed_idx+1}/{n_seeds_per_subset}")

                    run_dir = subset_dir / f"seed_{seed_idx}"

                    # Check if this run is already completed in the central tracker
                    if tracker_path is not None:
                        tracker_df = None
                        if Path(tracker_path).exists():
                            try:
                                tracker_df = pd.read_csv(tracker_path)
                                match = tracker_df[
                                    (tracker_df["experiment_name"] == experiment_name) &
                                    (tracker_df["subset_idx"] == subset_idx) &
                                    (tracker_df["seed_idx"] == seed_idx)
                                ]
                                if not match.empty:
                                    status = match.iloc[0]["status"]
                                    rmse_all = match.iloc[0].get("test_rmse_all_ts", match.iloc[0].get("rmse", float('inf')))
                                    print(f"  Skipping seed {seed_idx} - already in TRACKER CSV (Status: {status.upper()}).")
                                    seed_results.append({
                                        "subset_idx": subset_idx,
                                        "seed_idx": seed_idx,
                                        "status": status,
                                        "rmse": rmse_all,
                                    })
                                    continue
                            except Exception as e:
                                print(f"  Warning: Could not read tracker CSV: {e}")

                    # Check if this run is already completed locally
                    meta_path = run_dir / "run_meta.json"
                    if meta_path.exists():
                        try:
                            with open(meta_path, "r") as f:
                                existing_meta = json.load(f)
                            
                            status = existing_meta.get("status", "unknown")
                            rmse_all = existing_meta.get("test_rmse_all_ts", existing_meta.get("test_rmse", float('inf')))
                            rmse_train_ts = existing_meta.get("test_rmse_train_ts", float('inf'))
                            train_time = existing_meta.get("train_time_seconds", 0.0)
                            actual_steps = existing_meta.get("actual_steps_completed", 0)
                            training_failed = (status == "failed")
                            early_stopped = existing_meta.get("early_stopped", False)

                            print(f"  Skipping seed {seed_idx} - already completed (Status: {status.upper()}).")
                            
                            seed_results.append({
                                "subset_idx": subset_idx,
                                "seed_idx": seed_idx,
                                "status": status,
                                "rmse": rmse_all,
                                "test_rmse_train_ts": rmse_train_ts,
                                "train_time": train_time,
                                "steps_completed": actual_steps,
                                "early_stopped": early_stopped,
                            })
                            continue
                        except Exception as e:
                            print(f"  Warning: Could not read existing run_meta.json for seed {seed_idx}: {e}. Retraining.")
                    
                    # Atomic lock to claim the run and prevent duplicates in parallel execution
                    lock_path = base_path / f"lock_{subset_idx}_{seed_idx}.lock"
                    try:
                        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                    except FileExistsError:
                        # Check if stale (older than 3 hours)
                        try:
                            if time.time() - lock_path.stat().st_mtime > 10800:
                                print(f"  Removing stale lock file: {lock_path.name}")
                                os.remove(lock_path)
                                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                                os.close(fd)
                            else:
                                print(f"  Seed {seed_idx} of Subset {subset_idx} is currently locked by another worker. Skipping.")
                                continue
                        except Exception:
                            continue

                    if not subset_dir_created:
                        subset_dir.mkdir(parents=True, exist_ok=True)
                        with open(subset_dir / "subset_meta.json", "w") as f:
                            json.dump(subset_meta, f, indent=2)
                        subset_dir_created = True

                    run_dir.mkdir(parents=True, exist_ok=True)
                    training_failed = False
                    loss_history = []
                    val_losses = []
                    actual_steps = 0
                    train_time = 0.0
                    early_stopping_info = None
                    rmse_all = float('inf')
                    rmse_train_ts = float('inf')
                    eval_exclusion_info = {}
                    status = "unknown"
                    recovered_checkpoint = None
                    error_message = None
                    best_val_loss = float('nan')
                    best_train_loss = float('nan')

                    if self.is_xgboost:
                        try:
                            model = self._make_xgboost_model(hyper_parameters, seed=seed_idx)
                            loss_history, train_time = self._train_xgboost(
                                model, train_idx_actual, train_ys_actual, train_params_actual
                            )
                            # Save model
                            with open(run_dir / "model.pkl", "wb") as f:
                                pickle.dump(model, f)

                            # Evaluate
                            rmse_all, rmse_train_ts, eval_exclusion_info = self._get_RMSE_xgboost(model, test_ys, test_params, true_c_ys=self.true_ys[test_idx])
                            print(f"  XGBoost Seed {seed_idx} Test RMSE (pCA final): {rmse_all:.6f}")
                            status = "success"
                        except Exception as e:
                            error_message = str(e)
                            print(f"ERROR: XGBoost seed {seed_idx} failed: {error_message}")
                            training_failed = True
                            status = "failed"

                    else:
                        model_config = hyper_parameters.copy()
                        model_config["seed"] = seed_idx
                        try:
                            model = self.model_class(self.model_path, self.mask, model_config=model_config)
                        except Exception as e:
                            error_message = str(e)
                            print(f"ERROR: Could not create model for seed {seed_idx}: {error_message}")
                            training_failed = True
                            model = None

                        if not training_failed:
                            training_config = hyper_parameters.copy()
                            training_config["val_split"] = val_split
                            training_config["n_val_strains"] = n_val_strains
                            training_config["max_steps"] = max_steps
                            training_config["patience"] = patience
                            training_config["checkpoint_path"] = str(run_dir / "checkpoints")
                            training_config["checkpoint_freq"] = checkpoint_freq
                            training_config["print_freq"] = print_freq
                            training_config["timeout"] = timeout_seconds

                            try:
                                loss_history, early_stopping_info, train_time, best_val_loss, val_losses, best_train_loss = model.train(
                                    self.ts[self.ts_indices],
                                    train_ys_actual[:, self.ts_indices, :],
                                    train_params_actual,
                                    val_ys=val_ys_internal_data[:, self.ts_indices, :],
                                    val_params=val_params_internal_data,
                                    training_config=training_config,
                                )
                                actual_steps = len(loss_history)

                                if early_stopping_info is not None:
                                    reason = early_stopping_info.get("reason", "")
                                    if reason in ("diverged", "first_step_timeout", "nan_diverged"):
                                        print(f"  Seed {seed_idx} training '{reason}'. Attempting checkpoint recovery...")
                                        
                                        # Attempt checkpoint recovery
                                        recovered_model = None
                                        recovered_checkpoint = None
                                        checkpoint_dir = Path(training_config["checkpoint_path"])
                                        if checkpoint_dir.exists():
                                            all_cps = sorted(list(checkpoint_dir.glob("step_*")), 
                                                           key=lambda x: int(x.name.split("_")[1]), reverse=True)
                                            print(f"    Found {len(all_cps)} checkpoints for recovery.")
                                            
                                            for cp in all_cps:
                                                try:
                                                    temp_model = self.model_class.load(cp)
                                                    # Verify checkpoint produces finite predictions
                                                    test_idx_sample = test_idx[:min(5, len(test_idx))]
                                                    y0_sample = self.ys[test_idx_sample, 0, :]
                                                    p_sample = self.params[test_idx_sample]
                                                    
                                                    # Vmap the sample call
                                                    @jax.jit
                                                    def sample_predict(y0, p):
                                                        return temp_model(self.ts, y0, p)
                                                    
                                                    test_preds_sample = jax.vmap(sample_predict)(y0_sample, p_sample)
                                                    
                                                    if jnp.all(jnp.isfinite(test_preds_sample)):
                                                        recovered_model = temp_model
                                                        recovered_checkpoint = cp.name
                                                        print(f"    Recovered successfully from checkpoint: {recovered_checkpoint}")
                                                        break
                                                    else:
                                                        print(f"    Checkpoint {cp.name} contains NaNs, trying older one...")
                                                except Exception as cp_err:
                                                    error_message = f"Recovery failed: {str(cp_err)}"
                                                    print(f"    Failed to load checkpoint {cp.name}: {cp_err}")
                                                    continue
                                        
                                        if recovered_model is not None:
                                            model = recovered_model
                                            status = "recovered"
                                            training_failed = False # Treat as success-path for evaluation
                                        else:
                                            print(f"  Seed {seed_idx} recovery failed. Marking as FAILED.")
                                            training_failed = True
                                            status = "failed"
                                    else:
                                        status = "success"
                                        training_failed = False
                                else:
                                    status = "success"
                                    training_failed = False

                            except Exception as e:
                                error_message = str(e)
                                print(f"ERROR: Seed {seed_idx} training failed: {error_message}")
                                training_failed = True
                                status = "failed"
                                loss_history = []
                                actual_steps = 0
                                recovered_checkpoint = None

                        if not training_failed:
                            try:
                                rmse_all, rmse_train_ts, eval_exclusion_info = self.get_RMSE(model, test_ys, test_params, true_c_ys=self.true_ys[test_idx])
                                print(f"  Seed {seed_idx} Test RMSE (All ts): {rmse_all:.6f} | (Train ts): {rmse_train_ts:.6f}")
                                if eval_exclusion_info.get("eval_excluded_points", 0) > 0:
                                    print(f"  ({eval_exclusion_info['eval_excluded_points']}/{eval_exclusion_info['eval_total_points']} "
                                          f"entries excluded from RMSE due to non-finite predictions)")
                            except Exception as e:
                                error_message = str(e)
                                print(f"ERROR: Seed {seed_idx} evaluation failed: {error_message}")
                                rmse_all = float('inf')
                                rmse_train_ts = float('inf')
                                training_failed = True
                                
                            if np.isinf(rmse_all) or np.isnan(rmse_all):
                                status = "failed"
                                training_failed = True

                            try:
                                model.save(run_dir)
                            except Exception as e:
                                print(f"WARNING: Failed to save model seed {seed_idx}: {e}")

                    losses_subset.append(loss_history)
                    val_losses_subset.append(val_losses)

                 
                    run_meta = {
                        "subset_idx": subset_idx,
                        "seed_idx": seed_idx,
                        "status": status,
                        "error_message": error_message,
                        "recovered_checkpoint": recovered_checkpoint if status == "recovered" else None,
                        "train_indices": train_idx.tolist(),
                        "validation_indices": val_idx_internal.tolist(),
                        "test_indices": test_idx.tolist(),
                        "train_time_seconds": train_time,
                        "test_rmse": float(rmse_all),
                        "test_rmse_all_ts": float(rmse_all),
                        "test_rmse_train_ts": float(rmse_train_ts),
                        "n_train_total": n_train_strains,
                        "n_train_actual": len(train_idx_actual),
                        "n_validation": len(val_idx_internal),
                        "n_test": n_test_strains,
                        "actual_steps_completed": actual_steps,
                        "patience": patience,
                        "val_split": val_split,
                        "early_stopped": actual_steps < max_steps and not training_failed and not self.is_xgboost,
                        "timeout_seconds": timeout_seconds if not self.is_xgboost else None,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ts_indices": self.ts_indices.tolist() if hasattr(self.ts_indices, "tolist") else list(self.ts_indices),
                        "print_freq": print_freq,
                        "early_stopping_info": early_stopping_info,
                        "best_val_loss": float(best_val_loss),
                        "best_train_loss": float(best_train_loss),
                        "model_type": "XGBoost" if self.is_xgboost else str(self.model_class.__name__),
                        # Exclusion stats: how many prediction entries were non-finite and excluded from RMSE
                        "eval_total_points": eval_exclusion_info.get("eval_total_points", None),
                        "eval_excluded_points": eval_exclusion_info.get("eval_excluded_points", None),
                        "eval_excluded_fraction": eval_exclusion_info.get("eval_excluded_fraction", None),
                        "metabolite_indices": self.metabolite_indices if isinstance(self.metabolite_indices, list) else (self.metabolite_indices.tolist() if hasattr(self.metabolite_indices, "tolist") else self.metabolite_indices),
                    }

                    # Add hyperparameters that exist in the config 
                    for key in ["width_NODE", "depth_NODE", "dropout", "learning_rate", "batch_size",
                                "max_depth", "n_estimators"]:
                        if key in hyper_parameters:
                            run_meta[key] = hyper_parameters[key]

                    with open(run_dir / "run_meta.json", "w") as f:
                        json.dump(run_meta, f, indent=2)

                    # Log to central tracker
                    if tracker_path:
                        self._log_run_to_csv(tracker_path, experiment_name, run_meta)

                    # Save loss history 
                    if len(loss_history) > 0:
                        np.save(run_dir / "loss_history.npy", np.array(loss_history))
                        if len(val_losses) > 0:
                            np.save(run_dir / "val_loss_history.npy", np.array(val_losses))
                        data = {"step": np.arange(len(loss_history)), "loss": loss_history}
                        if len(val_losses) == len(loss_history):
                            data["val_loss"] = val_losses
                        pd.DataFrame(data).to_csv(run_dir / "loss_history.csv", index=False)

                        # Plot individual loss history
                        if make_plots:
                            self._plot_individual_loss(loss_history, val_losses, run_dir, seed_idx, subset_idx)


                    seed_results.append({
                        "subset_idx": subset_idx,
                        "seed_idx": seed_idx,
                        "status": status,
                        "recovered_checkpoint": recovered_checkpoint if status == "recovered" else None,
                        "rmse": rmse_all,
                        "test_rmse_train_ts": rmse_train_ts,
                        "train_time": train_time,
                        "steps_completed": actual_steps,
                        "early_stopped": actual_steps < max_steps and not training_failed and not self.is_xgboost,
                    })

                    # Per-run plots
                    if not training_failed:
                        try:
                            self.plot_results_per_run(model, run_dir, plot=make_plots)
                        except Exception as e:
                            print(f"WARNING: plot_results_per_run failed for seed {seed_idx}: {e}")

                    _rmse_str = f" | RMSE: {rmse_all:.6f}" if not training_failed else " | RMSE: N/A"
                    _status_str = status.upper()
                    if status == "recovered":
                        _status_str += f" ({recovered_checkpoint})"
                    print(f"  Seed {seed_idx} Summary: {_status_str}"
                        f" | Steps: {actual_steps} | Time: {train_time:.1f}s{_rmse_str}")

                    # Aggressively clear memory to prevent hoarding RAM over 25 iterations
                    del model
                    gc.collect()

                # Clear JAX compiled code cache once per subset (not per seed) to free
                # LLVM mmap allocations. This prevents vm.max_map_count exhaustion
                # while limiting recompilation to n_subsets times (not n_total_models).
                try:
                    jax.clear_caches()
                except Exception:
                    pass
                gc.collect()

        except KeyboardInterrupt:
            print("  INTERRUPTED BY USER - Stopping all experiments.")
            raise

        return

   

    def _plot_individual_loss(self, loss_history, val_losses, run_dir, seed_idx, subset_idx):
        """Plot the training and validation loss for a single run."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        loss_history_np = np.array(loss_history)
        steps = np.arange(len(loss_history_np))
        
        # Training loss
        ax.plot(steps, loss_history_np, label="Training Loss", color="#1f77b4", linewidth=1.5, alpha=0.9)
        
        # Validation loss
        if len(val_losses) > 0 and len(val_losses) == len(loss_history_np):
            val_losses_np = np.array(val_losses)
            # Filter out negative values if they exist 
            mask = val_losses_np >= 0
            if np.any(mask):
                ax.plot(steps[mask], val_losses_np[mask], label="Validation Loss", 
                        color="#975215", linewidth=1.5, linestyle="--", alpha=0.8)
        
        ax.set_yscale("log")
        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel("Loss (Log Scale)", fontsize=12)
        ax.set_title(f"Run Loss History: Subset {subset_idx}, Seed {seed_idx}", fontsize=14, pad=15)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(frameon=True, loc="upper right", fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(run_dir / "loss_history.png", dpi=150, bbox_inches="tight")
        plt.close()


    def plot_results_per_run(self, model, run_dir, plot=True):
        """ Gets all train and test predictions and saves them to results.json and predictions.npz. Optionally plots results."""
        with open(run_dir / "run_meta.json", "r") as f:
            run_meta = json.load(f)

        test_indices = run_meta["test_indices"]
        train_indices = [i for i in run_meta["train_indices"] if i not in run_meta["validation_indices"]]

        test_res = self.get_preds(model, test_indices)
        train_res = self.get_preds(model, train_indices)

        results = {
            "ModelType": "XGBoost" if self.is_xgboost else str(self.model_class.__name__),
            "RunID": run_dir.name,
            "RMSE_All_Species_All_TS": test_res["RMSE_All_Species_All_TS"],
            "RMSE_All_Species_Train_TS": test_res["RMSE_All_Species_Train_TS"],
            "NRMSE_All_Species_Train_TS": test_res["NRMSE_All_Species_Train_TS"],
            "RMSE_pCA_Final": test_res["RMSE_pCA_Final"],
            "NRMSE_pCA_Final": test_res["NRMSE_pCA_Final"],
            "RMSE_pCA_All": test_res["RMSE_pCA_All"],
            "NRMSE_All_Species": test_res["NRMSE_All_Species"],
            "final_pCA_preds": test_res["final_pCA_preds"],
            "TestIndices": test_res["Indices"],
            "NRMSE_per_species": test_res["NRMSE_per_species"],
            "rmse_per_strain": test_res["rmse_per_strain"],
            "nrmse_per_strain": test_res["nrmse_per_strain"],
            "ts_indices": self.ts_indices,
            "ExclusionInfo": test_res["ExclusionInfo"],

            "Train_RMSE_All_Species_All_TS": train_res["RMSE_All_Species_All_TS"],
            "Train_RMSE_All_Species_Train_TS": train_res["RMSE_All_Species_Train_TS"],
            "Train_NRMSE_All_Species_Train_TS": train_res["NRMSE_All_Species_Train_TS"],
            "Train_RMSE_pCA_Final": train_res["RMSE_pCA_Final"],
            "Train_NRMSE_pCA_Final": train_res["NRMSE_pCA_Final"],
            "Train_RMSE_pCA_All": train_res["RMSE_pCA_All"],
            "Train_NRMSE_All_Species": train_res["NRMSE_All_Species"],
            "Train_final_pCA_preds": train_res["final_pCA_preds"],
            "TrainIndices": train_res["Indices"],
            "Train_NRMSE_per_species": train_res["NRMSE_per_species"],
            "Train_rmse_per_strain": train_res["rmse_per_strain"],
            "Train_nrmse_per_strain": train_res["nrmse_per_strain"],
            "Train_ExclusionInfo": train_res["ExclusionInfo"],
        }

        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

        # Save arrays efficiently
        np.savez_compressed(
            run_dir / "predictions.npz", 
            Predictions=test_res["Predictions"], 
            Train_Predictions=train_res["Predictions"]
        )
        
        # Save ground truths at the subset level if they don't exist yet
        subset_dir = run_dir.parent
        gt_path = subset_dir / "ground_truths.npz"
        if not gt_path.exists():
            np.savez_compressed(
                gt_path,
                GroundTruths=test_res["GroundTruths"],
                Train_GroundTruths=train_res["GroundTruths"]
            )
            
        # Add arrays back to results dict temporarily for downstream plotting logic in this function
        results["Predictions"] = test_res["Predictions"]
        results["GroundTruths"] = test_res["GroundTruths"]
        results["Train_Predictions"] = train_res["Predictions"]
        results["Train_GroundTruths"] = train_res["GroundTruths"]

        if plot: 
            # Scatter plot pCA final value
            test_preds_pCA = results["final_pCA_preds"]
            test_truths_pCA = [float(np.array(t)[-1, self.pCA_index]) for t in results["GroundTruths"]]
            train_preds_pCA = results["Train_final_pCA_preds"]
            train_truths_pCA = [float(np.array(t)[-1, self.pCA_index]) for t in results["Train_GroundTruths"]]

            plt.scatter(train_truths_pCA, train_preds_pCA, alpha=0.5, marker="x", color="orange",
                        label=f"Train NRMSE: {results['Train_NRMSE_pCA_Final']:.1f}%")
            plt.scatter(test_truths_pCA, test_preds_pCA, alpha=0.7, marker="o", color="blue",
                        label=f"Test  NRMSE: {results['NRMSE_pCA_Final']:.1f}%")

            all_vals = test_preds_pCA + test_truths_pCA + train_preds_pCA + train_truths_pCA
            min_val, max_val = min(all_vals), max(all_vals)
            plt.plot([min_val, max_val], [min_val, max_val], "k--")
            plt.xlabel("Ground Truth")
            plt.ylabel("Prediction")
            plt.legend(loc="best", fontsize="small")
            plt.savefig(run_dir / "scatter_plot.png", dpi=150, bbox_inches="tight")
            plt.close()

            # Per-strain bar plots 
            sorted_train_idx = np.argsort(results["Train_rmse_per_strain"])
            sorted_test_idx = np.argsort(results["rmse_per_strain"])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            axes[0].bar(range(len(sorted_train_idx)),
                        np.array(results["Train_rmse_per_strain"])[sorted_train_idx],
                        color="orange", alpha=0.7)
            axes[0].set_xlabel("Strain Rank")
            axes[0].set_ylabel("RMSE per Strain")
            axes[0].set_title("Training Set RMSE per Strain")

            axes[1].bar(range(len(sorted_test_idx)),
                        np.array(results["rmse_per_strain"])[sorted_test_idx],
                        color="blue", alpha=0.7)
            axes[1].set_xlabel("Strain Rank")
            axes[1].set_title("Test Set RMSE per Strain")

            plt.suptitle("RMSE per Strain (Train vs Test)", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(run_dir / "rmse_per_strain_subplots.png", dpi=150)
            plt.close()

            # Trajectory plots
            self.plot_trajectories_by_error(
                results, ts=self.ts, ts_indices=self.ts_indices,
                species_names=self.species_names, pCA_index=self.pCA_index,
                set_name="Train", save_path=run_dir,
            )
            self.plot_trajectories_by_error(
                results, ts=self.ts, ts_indices=self.ts_indices,
                species_names=self.species_names, pCA_index=self.pCA_index,
                set_name="Test", save_path=run_dir,
            )

            # pCA trajectories train vs test
            train_preds_np = np.array(results["Train_Predictions"])
            train_truths_np = np.array(results["Train_GroundTruths"])
            test_preds_np = np.array(results["Predictions"])
            test_truths_np = np.array(results["GroundTruths"])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

            ax = axes[0]
            for i in range(len(train_preds_np)):
                line, = ax.plot(self.ts, train_preds_np[i, :, self.pCA_index], alpha=0.8)
                color = line.get_color()
                ax.scatter(self.ts[self.ts_indices], train_truths_np[i, self.ts_indices, self.pCA_index],
                        color="red", marker="x", s=50, alpha=0.9)
                unseen = np.setdiff1d(np.arange(len(self.ts)), self.ts_indices)
                ax.scatter(self.ts[unseen], train_truths_np[i, unseen, self.pCA_index],
                        color=color, marker="x", s=45, alpha=0.7)
            ax.set_title("Training Strains")
            ax.set_xlabel("Time")
            ax.set_ylabel("Product Concentration")

            ax = axes[1]
            for i in range(len(test_preds_np)):
                line, = ax.plot(self.ts, test_preds_np[i, :, self.pCA_index], alpha=0.8)
                color = line.get_color()
                ax.scatter(self.ts[self.ts_indices], test_truths_np[i, self.ts_indices, self.pCA_index],
                        color="red", marker="x", s=50, alpha=0.9)
                unseen = np.setdiff1d(np.arange(len(self.ts)), self.ts_indices)
                ax.scatter(self.ts[unseen], test_truths_np[i, unseen, self.pCA_index],
                        color=color, marker="x", s=45, alpha=0.7)
            ax.set_title("Test Strains")
            ax.set_xlabel("Time")

            handles = [
                plt.Line2D([0], [0], marker="x", color="red", linestyle="", markersize=8, label="GT (training points)"),
                plt.Line2D([0], [0], marker="x", color="black", linestyle="", markersize=8, label="GT (unseen points)"),
                plt.Line2D([0], [0], color="black", label="Prediction"),
            ]
            axes[1].legend(handles=handles, loc="best")
            plt.tight_layout()
            plt.savefig(run_dir / "product_trajectories_train_vs_test.png", dpi=150)
            plt.close()

            return

    def plot_trajectories_by_error(self, results, ts, ts_indices, species_names, pCA_index, set_name="Train", save_path=None):
        """Plot predicted trajectories vs ground truth for strains grouped by RMSE percentiles."""

        if set_name == "Train":
            rmse_per_strain = np.array(results["Train_rmse_per_strain"])
            preds = np.array(results["Train_Predictions"])
            truths = np.array(results["Train_GroundTruths"])
        else:
            rmse_per_strain = np.array(results["rmse_per_strain"])
            preds = np.array(results["Predictions"])
            truths = np.array(results["GroundTruths"])

        percentiles = np.arange(0, 110, 10)

        n_strains = len(rmse_per_strain)
        sorted_indices = np.argsort(rmse_per_strain)
        strain_indices = [sorted_indices[int(p / 100 * (n_strains - 1))] for p in percentiles]

        n_subplots = len(percentiles)
        n_cols = 5
        n_rows = int(np.ceil(n_subplots / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, strain_idx in enumerate(strain_indices):
            ax = axes[i]
            ax.set_title(f"{percentiles[i]}th percentile\nStrain {strain_idx} rmse: {rmse_per_strain[strain_idx]:.4f}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")

            for k, species in enumerate(species_names):
                color = f"C{k}"
                ax.plot(ts, preds[strain_idx, :, k], linestyle="--", color=color, alpha=0.8, label=species)
                ax.scatter(ts[ts_indices], truths[strain_idx, ts_indices, k], color=color, marker="o", s=40, alpha=0.5)

            if i == 0:
                ax.legend(loc="best", fontsize="small")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path / f"{set_name}_trajectories_percentiles.png", dpi=150)
        plt.close()


    def calculate_evaluation_metrics(self, preds, truths, set_name="test"):
        """
        Standardized calculation of RMSE and NRMSE metrics.
        preds: (n_samples, n_times, n_species)
        truths: (n_samples, n_times, n_species)
        """
        # Skip t=0 (initial condition) for evaluation
        data_slice = truths[:, 1:, :]   # (n_strains, n_times-1, n_species)
        pred_slice = preds[:, 1:, :]

        finite_mask = jnp.isfinite(data_slice) & jnp.isfinite(pred_slice)
        total_points  = int(finite_mask.size)
        excluded_points = int((~finite_mask).sum())

        exclusion_info = {
            f"{set_name}_total_points": total_points,
            f"{set_name}_excluded_points": excluded_points,
            f"{set_name}_excluded_fraction": round(excluded_points / total_points, 6) if total_points > 0 else 1.0,
  
            "eval_total_points": total_points,
            "eval_excluded_points": excluded_points,
            "eval_excluded_fraction": round(excluded_points / total_points, 6) if total_points > 0 else 1.0,
        }

        # 1. All Species All TS (timesteps) RMSE
        if excluded_points < total_points:
            diff_all = jnp.where(finite_mask, data_slice - pred_slice, 0.0)
            mse_all  = jnp.sum(jnp.square(diff_all)) / jnp.sum(finite_mask)
            rmse_all = float(jnp.sqrt(mse_all))
            
            # Normalized All Species All TS
            err_sq_per_species = jnp.sum(jnp.square(diff_all), axis=(0, 1)) / jnp.sum(finite_mask, axis=(0, 1))
            rmse_per_species = jnp.sqrt(err_sq_per_species)
            nrmse_per_species = (rmse_per_species / jnp.array(self.metabolite_ranges)) * 100
            nrmse_all_species = float(jnp.mean(nrmse_per_species))
        else:
            rmse_all = float("nan")
            nrmse_per_species = [float("nan")] * len(self.metabolite_ranges)
            nrmse_all_species = float("nan")

        # 2. pCA Specific Metrics
        pCA_data = data_slice[:, :, self.pCA_index]
        pCA_pred = pred_slice[:, :, self.pCA_index]
        pCA_mask = finite_mask[:, :, self.pCA_index]
        n_finite_pCA = int(jnp.sum(pCA_mask))
        
        if n_finite_pCA > 0:
            diff_pCA = jnp.where(pCA_mask, pCA_data - pCA_pred, 0.0)
            rmse_pCA_all = float(jnp.sqrt(jnp.sum(jnp.square(diff_pCA)) / n_finite_pCA))
            
            # Final TS pCA
            pCA_final_data = truths[:, -1, self.pCA_index]
            pCA_final_pred = preds[:, -1, self.pCA_index]
            pCA_final_mask = jnp.isfinite(pCA_final_data) & jnp.isfinite(pCA_final_pred)
            if jnp.any(pCA_final_mask):
                rmse_pCA_final = float(jnp.sqrt(jnp.mean(jnp.square(pCA_final_data - pCA_final_pred)[pCA_final_mask])))
            else:
                rmse_pCA_final = float("nan")
        else:
            rmse_pCA_all = float("nan")
            rmse_pCA_final = float("nan")

        # 3. Train TS Metrics (All Species)
        ts_pos = [int(idx) if idx >= 0 else int(idx + len(self.ts)) for idx in np.array(self.ts_indices).tolist()]
        ts_indices_no_zero = [idx for idx in ts_pos if idx != 0] # skipping t=0
        
        if len(ts_indices_no_zero) > 0:
            data_train = truths[:, ts_indices_no_zero, :]
            pred_train = preds[:, ts_indices_no_zero, :]
            mask_train = jnp.isfinite(data_train) & jnp.isfinite(pred_train)
            n_finite_train = int(jnp.sum(mask_train))
            if n_finite_train > 0:
                diff_train = jnp.where(mask_train, data_train - pred_train, 0.0)
                rmse_train_ts = float(jnp.sqrt(jnp.sum(jnp.square(diff_train)) / n_finite_train))
                
                # pCA Train TS
                pCA_train_mask = mask_train[:, :, self.pCA_index]
                if jnp.any(pCA_train_mask):
                    diff_pCA_train = jnp.where(pCA_train_mask, data_train[:, :, self.pCA_index] - pred_train[:, :, self.pCA_index], 0.0)
                    rmse_pCA_train_ts = float(jnp.sqrt(jnp.sum(jnp.square(diff_pCA_train)) / jnp.sum(pCA_train_mask)))
                else:rmse_pCA_train_ts = float("nan")

                # Normalized Train TS (All Species)
                err_sq_per_species_train = jnp.sum(jnp.square(diff_train), axis=(0, 1)) / jnp.sum(mask_train, axis=(0, 1))
                rmse_per_species_train = jnp.sqrt(err_sq_per_species_train)
                nrmse_per_species_train = (rmse_per_species_train / jnp.array(self.metabolite_ranges)) * 100
                nrmse_train_ts = float(jnp.mean(nrmse_per_species_train))
            else:
                rmse_train_ts = float("nan")
                rmse_pCA_train_ts = float("nan")
                nrmse_train_ts = float("nan")
        else:
            rmse_train_ts = float("nan")
            rmse_pCA_train_ts = float("nan")
            nrmse_train_ts = float("nan")


        return {
            "RMSE_All_Species_All_TS": rmse_all,
            "NRMSE_All_Species_All_TS": nrmse_all_species,
            "RMSE_All_Species_Train_TS": rmse_train_ts,
            "NRMSE_All_Species_Train_TS": nrmse_train_ts,
            "RMSE_pCA_All_TS": rmse_pCA_all,
            "RMSE_pCA_Train_TS": rmse_pCA_train_ts,
            "RMSE_pCA_Final": rmse_pCA_final,
            "NRMSE_pCA_Final": (rmse_pCA_final / self.metabolite_ranges[self.pCA_index]) * 100 if not jnp.isnan(rmse_pCA_final) else float("nan"),
            "NRMSE_per_species": list(nrmse_per_species),
            "ExclusionInfo": exclusion_info
        }

    def get_preds(self, model, indices):
        indices_arr = np.array(indices)
        batch_y0 = self.ys[indices_arr, 0, :]
        batch_p = self.params[indices_arr]
        batch_y_true = self.true_ys[indices_arr]

        if self.is_xgboost:
            # Predict trajectories for EVERY timepoint in self.ts for consistent evaluation
            y_preds_traj = []
            for t_val in self.ts:
                t_arr = np.full((len(batch_p), 1), t_val)
                X_t = np.concatenate([batch_p, t_arr], axis=1)
                y_preds_traj.append(model.predict(X_t, y0s=batch_y0))
            
            y_preds_jax = jnp.stack(y_preds_traj, axis=1)
            y_preds_np = np.array(y_preds_jax)
        else: 
            vmapped_model = jax.vmap(lambda y0, p: model(self.ts, y0, p))
            y_preds_jax = vmapped_model(batch_y0, batch_p)
            y_preds_np = np.array(y_preds_jax)

        
        metrics = self.calculate_evaluation_metrics(y_preds_jax, batch_y_true)
        
        # Additional per-strain metrics for visualization
        finite_mask_np = np.array(metrics["ExclusionInfo"].get("finite_mask", np.isfinite(y_preds_np[:, 1:, :]) & np.isfinite(np.array(batch_y_true)[:, 1:, :])))
        diff_sq_np = (y_preds_np[:, 1:, :] - np.array(batch_y_true)[:, 1:, :]) ** 2
        
        # Calculate mean square error per strain using the finite mask
        # Shape of diff_sq_np: (n_strains, n_times-1, n_metabolites)
        # Sum over times and metabolites for each strain
        mse_per_strain = np.sum(np.where(finite_mask_np, diff_sq_np, 0.0), axis=(1, 2)) / np.sum(finite_mask_np, axis=(1, 2))
        rmse_per_strain_all_ts = np.sqrt(mse_per_strain)
        
        # Consistent normalization
        nrmse_per_strain_all_ts = rmse_per_strain_all_ts / np.mean(self.metabolite_ranges) * 100

        res = {
            "RMSE_pCA_Final": metrics["RMSE_pCA_Final"],
            "NRMSE_pCA_Final": metrics["NRMSE_pCA_Final"],
            "RMSE_pCA_All": metrics["RMSE_pCA_All_TS"],
            "RMSE_All_Species_All_TS": metrics["RMSE_All_Species_All_TS"],
            "NRMSE_All_Species": metrics["NRMSE_All_Species_All_TS"],
            "RMSE_pCA_Train_TS": metrics["RMSE_pCA_Train_TS"],
            "RMSE_All_Species_Train_TS": metrics["RMSE_All_Species_Train_TS"],
            "NRMSE_All_Species_Train_TS": metrics["NRMSE_All_Species_Train_TS"],
            "NRMSE_per_species": metrics["NRMSE_per_species"],
            "ExclusionInfo": metrics["ExclusionInfo"],
            
            "final_pCA_preds": list(np.array(y_preds_jax[:, -1, self.pCA_index])),
            "GroundTruths": list(np.array(batch_y_true)),
            "Predictions": list(np.array(y_preds_jax)),
            "Indices": list(indices),
            "ts_indices": list(self.ts_indices),
            "rmse_per_strain": list(rmse_per_strain_all_ts),
            "nrmse_per_strain": list(nrmse_per_strain_all_ts),
        }
        return res

    def get_RMSE(self, model, c_ys, c_params, true_c_ys=None):
        if true_c_ys is None: true_c_ys = c_ys
        """Standardized RMSE for ODE models."""
        y0s = c_ys[:, 0, :]
        preds = []
        for i in range(c_ys.shape[0]):
            preds.append(model(self.ts, y0s[i], c_params[i]))
        preds = jnp.stack(preds)
        
        metrics = self.calculate_evaluation_metrics(preds, true_c_ys)
        return metrics["RMSE_All_Species_All_TS"], metrics["RMSE_All_Species_Train_TS"], metrics["ExclusionInfo"]

    def _get_RMSE_xgboost(self, model, c_ys, c_params, true_c_ys=None):
        if true_c_ys is None: true_c_ys = c_ys
        """Standardized RMSE for XGBoost (now using trajectories)."""
        y0s = c_ys[:, 0, :]
        preds_traj = []
        for t_val in self.ts:
            t_arr = np.full((len(c_params), 1), t_val)
            X_t = np.concatenate([c_params, t_arr], axis=1)
            preds_traj.append(model.predict(X_t, y0s=y0s))
        
        preds = jnp.stack(preds_traj, axis=1)
        metrics = self.calculate_evaluation_metrics(preds, true_c_ys)
        return metrics["RMSE_All_Species_All_TS"], metrics["RMSE_All_Species_Train_TS"], metrics["ExclusionInfo"]

    def _log_run_to_csv(self, tracker_path, experiment_name, run_meta):
        """Append a single run's metadata and results to a central CSV tracker, handling dynamic columns."""
        tracker_path = Path(tracker_path)
        
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": experiment_name,
        }
        
       
        exclude_keys = ["fixed_Hybrid_RMSE", "fixed_NODE_RMSE", "max_depth", "n_estimators"]
        for k, v in run_meta.items():
            if k in exclude_keys:
                continue
            if isinstance(v, (list, np.ndarray, jnp.ndarray)):
                if k in ["train_indices", "validation_indices", "test_indices"]:
                    continue # Skip large index lists
                row[k] = str(list(v)) if hasattr(v, "tolist") else str(v)
            else:
                row[k] = v
                
        # Ensure directory exists
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe Append to CSV using a lock file
        lock_path = tracker_path.with_suffix(".lock")
        with open(lock_path, "w") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                
                if tracker_path.exists():
                    try:
                        # Read existing header
                        df_existing = pd.read_csv(tracker_path)
                        existing_cols = df_existing.columns.tolist()
                        
                        # Check for new columns in the new row
                        new_cols = [c for c in row.keys() if c not in existing_cols]
                        
                        if new_cols:
                           
                            for c in new_cols:
                                df_existing[c] = None
                            
                            df_new = pd.DataFrame([row])
                            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                            df_combined.to_csv(tracker_path, index=False)
                        else:
                           
                            df_new = pd.DataFrame([row])
                            # Reorder columns to match existing header
                            df_new = df_new.reindex(columns=existing_cols)
                            df_new.to_csv(tracker_path, mode='a', header=False, index=False)
                    except Exception as e:
                        # Fallback if CSV is malformed 
                        print(f"WARNING: Tracker CSV error: {e}. Attempting recovery...")
                        try:
                            # Try to read raw and fix it
                            df_raw = pd.read_csv(tracker_path, on_bad_lines='warn') 
                       
                            df_new = pd.DataFrame([row])
                            df_combined = pd.concat([df_raw, df_new], ignore_index=True)
                            df_combined.to_csv(tracker_path, index=False)
                        except:
                            
                            print("CRITICAL: Could not recover tracker CSV automatically.")
                else:
                    # Create new file
                    df = pd.DataFrame([row])
                    df.to_csv(tracker_path, mode='w', header=True, index=False)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

 