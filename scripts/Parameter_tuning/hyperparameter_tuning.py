import itertools
import pandas as pd
import jax.numpy as jnp
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import gc
import jax
import signal
import sys


class HyperparameterTuner:
    def __init__(self, model_class, base_params: Dict[str, Any], param_grid: Dict[str, List[Any]]):
  
        self.model_class = model_class
        self.base_params = base_params
        self.param_grid = param_grid
        self.results = []
        
    def save_results(self, output_path: Path):
        """Save results to a CSV file using an fcntl lock for parallel safety."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_csv = output_path / "tuning_results.csv"
        lock_file_path = output_path / "tuning_results.lock"
        
        import os
        import time
        import fcntl
        
        with open(lock_file_path, 'w') as lock_f:
            try:
              
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                
                existing_results_map = {}
                
                # Load existing CSV
                if summary_csv.exists():
                    try:
                        import pandas as pd
                        df_old = pd.read_csv(summary_csv)
                        for _, row in df_old.iterrows():
                            entry = row.to_dict()
                            key = (int(entry.get("config_id", -1)), int(entry.get("run_id", -1)))
                            existing_results_map[key] = entry
                    except Exception as e:
                        print(f"  Warning: Could not merge with existing CSV: {e}")

                # Re-scan for new ones not yet in CSV in case of missing csv file
                for config_path in sorted(output_path.glob("config_*")):
                    if not config_path.is_dir(): continue
                    try:
                        c_id = int(config_path.name.split('_')[1])
                        for run_path in sorted(config_path.glob("run_*")):
                            if not run_path.is_dir(): continue
                            r_id = int(run_path.name.split('_')[1])
                            meta_path = run_path / "run_meta.json"
                            if meta_path.exists():
                                import json
                                with open(meta_path) as f:
                                    meta = json.load(f)
                                config_info = meta.get("config", {})
                                entry = {
                                    "config_id": c_id, "run_id": r_id,
                                    "status": meta.get("status", "unknown"),
                                    "test_rmse": meta.get("test_rmse", float("inf")),
                                    "val_loss": meta.get("final_val_loss", float("inf")),
                                    "training_time": meta.get("train_time_seconds", None),
                                    "iterations": meta.get("actual_steps_completed", None),
                                    "stopping_condition": meta.get("stopping_condition", "unknown"),
                                    "error": meta.get("error", ""),
                                    "output_dir": str(run_path)
                                }
                                for k, v in config_info.items(): entry[k] = v
                                existing_results_map[(c_id, r_id)] = entry
                    except: continue

           
                for res in self.results:
                    c_id = int(res.get("config_id", -1))
                    r_id = int(res.get("run_id", -1))
                    if c_id != -1 and r_id != -1:
                        key = (c_id, r_id)
                        entry = res.copy()
                        if "config" in entry:
                            config_info = entry.pop("config")
                            for k, v in config_info.items(): entry[k] = v
                        existing_results_map[key] = entry
                        
                if not existing_results_map: 
                    import pandas as pd
                    return pd.DataFrame()

                all_results = list(existing_results_map.values())
                import pandas as pd
                df = pd.DataFrame(all_results)
                if "config_id" in df.columns:
                    df = df[(df["config_id"] != -1) & (df["run_id"] != -1)]
                    df.sort_values(["config_id", "run_id"], inplace=True)
                    
                temp_csv = output_path / f"tuning_results_temp_{os.getpid()}.csv"
                df.to_csv(temp_csv, index=False)
                temp_csv.replace(summary_csv)
                return df
                
            finally:
           
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def load_existing_results(self, output_path: Path, silent=False):
        """Scans output directory (and summary CSV) for existing runs and populates results list."""
        existing_configs = {} # Map config_tuple -> list of run_ids
        existing_run_ids = []
        self.results = [] 
        
        if not output_path.exists():
            return {}, []

        if not silent:
            print(f"Scanning {output_path} for existing global results...")
        
       
        summary_csv = output_path / "tuning_results.csv"
        if summary_csv.exists():
            try:
                df = pd.read_csv(summary_csv)
                if not df.empty:
                    if not silent:
                        print(f"  Found {len(df)} results in current {summary_csv.name}")
                    for _, row in df.iterrows():
                        entry = row.to_dict()
                        c_id = int(entry.get("config_id", -1))
                        r_id = int(entry.get("run_id", -1))
                        
                        
                        config = {}
                        for k in self.param_grid.keys():
                            if k in entry:
                                config[k] = entry[k]
                        
                        config_tuple = tuple(sorted(config.items()))
                        if config_tuple not in existing_configs:
                            existing_configs[config_tuple] = []
                        if r_id not in existing_configs[config_tuple]:
                            existing_configs[config_tuple].append(r_id)
                        
                        # Add to the global results pool
                        res_entry = {
                            "config_id": c_id,
                            "run_id": r_id,
                            "status": entry.get("status", "unknown"),
                            "test_rmse": entry.get("test_rmse", float("inf")),
                            "val_loss": entry.get("val_loss", float("inf")),
                            "training_time": entry.get("training_time", None),
                            "iterations": entry.get("iterations", None),
                            "stopping_condition": entry.get("stopping_condition", "unknown"),
                            "config": config,
                        }
                        self.results.append(res_entry)
            except Exception as e:
                print(f"  Warning: Could not read {summary_csv.name}: {e}")

        
        for config_path in sorted(output_path.glob("config_*")):
            if not config_path.is_dir(): continue
            try:
                c_id_str = config_path.name.split('_')[1]
                if not c_id_str.isdigit(): continue
                c_id = int(c_id_str)
                
                for run_path in sorted(config_path.glob("run_*")):
                    if not run_path.is_dir(): continue
                    r_id_str = run_path.name.split('_')[1]
                    if not r_id_str.isdigit(): continue
                    r_id = int(r_id_str)
                    
                    meta_path = run_path / "run_meta.json"
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta = json.load(f)
                        
                        status = meta.get("status", "unknown")
                        # Only skip if the run was actually completed with a terminal status
                        terminal_statuses = ["success", "failed", "recovered", "checkpoint_recovered"]
                        if status not in terminal_statuses:
                            continue

                        config = meta.get("config", {})
                        config_tuple = tuple(sorted(config.items()))
                        
                        if config_tuple not in existing_configs:
                            existing_configs[config_tuple] = []
                        
                        if r_id not in existing_configs[config_tuple]:
                            existing_configs[config_tuple].append(r_id)
                            
                            res_entry = {
                                "config_id": c_id,
                                "run_id": r_id,
                                "status": meta.get("status", "unknown"),
                                "test_rmse": meta.get("test_rmse", float("inf")),
                                "val_loss": meta.get("final_val_loss", float("inf")),
                                "training_time": meta.get("train_time_seconds", None),
                                "iterations": meta.get("actual_steps_completed", None),
                                "stopping_condition": meta.get("stopping_condition", "unknown"),
                                "config": config,
                                "output_dir": str(run_path),
                            }
                            self.results.append(res_entry)
            except Exception:
                continue

        print(f"Total results loaded for resuming: {len(self.results)}")
        return existing_configs, existing_run_ids


    def reconstruct_csv(self, output_dir: str):
        output_path = Path(output_dir)
        self.load_existing_results(output_path)
        return self.save_results(output_path)

        
    def _wait_for_batch_completion(self, output_path, end_idx, n_runs_req):
        """Wait for all configs up to end_idx to have a run_meta or summary file."""
        import time
        from collections import defaultdict
        print(f"Waiting for batch completion (up to Config {end_idx})...")
        
        last_incomplete = -1
        last_progress_time = time.time()
        
        while True:
            self.load_existing_results(output_path, silent=True)
            
            completed_runs = defaultdict(int)
            for r in self.results:
                if r.get("status") in ["success", "failed"]:
                    c_id = r.get("config_id")
                    if c_id is not None and c_id <= end_idx:
                         completed_runs[c_id] += 1
            
            incomplete = 0
            for i in range(end_idx + 1):
                if completed_runs[i] < n_runs_req:
                    incomplete += 1

            if incomplete == 0:
                print("Batch complete.")
                break
                
            if incomplete != last_incomplete:
                last_incomplete = incomplete
                last_progress_time = time.time()
                
            if time.time() - last_progress_time > 3600:
                print(f"\nWARNING: Waited 1 hour with no new runs completing!")
                print(f"Assuming workers for the {incomplete} missing configurations have crashed.")
                print(f"Breaking barrier to proceed with available configurations.\n")
                break
                
            print(f"Batch incomplete: {incomplete} configs still missing runs. Waiting 30s...")
            time.sleep(30)

    def tune(self, ts, ys, params, print_freq=50, max_steps=1000, patience=20, output_dir="tuning_results", n_iter=None, n_runs=1, seed=42, checkpoint_freq=100, timeout=None, use_hill_climbing=False, hill_climbing_steps=5, n_train_pure=250, n_val_pure=50):

        
        val_split = n_val_pure / (n_train_pure + n_val_pure)
        n_train_total = n_train_pure + n_val_pure
        
        keys = list(self.param_grid.keys())
        
    
        total_combinations = 1
        for k in keys:
             total_combinations *= len(self.param_grid[k])
        
        import random
        rng = random.Random(seed)
        
       
        def handle_sigterm(signum, frame):
            print("\n!!! CAUGHT SIGTERM (SLURM LIMIT) - Saving results before exit !!!")
            self.save_results(output_path)
     
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_sigterm)

        combinations = []
        
        if n_iter is None:
            # Full Grid Search
            print(f"Starting Grid Search with {total_combinations} combinations.")
            values = [self.param_grid[k] for k in keys]
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            # Random Search
            print(f"Starting Random Search: sampling {n_iter} candidates from {total_combinations} possibilities.")
            
            seen_configs = set()
            attempts = 0
            # Safety break to prevent infinite loop if n_iter approaches total_combinations
            max_attempts = n_iter * 10 
            
            while len(combinations) < n_iter and attempts < max_attempts:
                attempts += 1
                config = {}
                for k in keys:
                    config[k] = rng.choice(self.param_grid[k])
                
                # Make hashable to check for duplicates
                config_tuple = tuple(sorted(config.items()))
                
                if config_tuple not in seen_configs:
                    seen_configs.add(config_tuple)
                    combinations.append(config)
            
            if len(combinations) < n_iter:
                print(f"Warning: Could only find {len(combinations)} unique combinations (requested {n_iter}).")
            
        print(f"Selected {len(combinations)} unique combinations.")

        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Scan for existing runs and sync CSV
        existing_configs_map, existing_run_ids = self.load_existing_results(output_path)
        self.save_results(output_path)
        
        # Pull best loss from existing results
        best_loss = float('inf')
        best_config = None
        for r in self.results:
            metric = r.get("test_rmse", r.get("val_loss", float('inf')))
            if metric < best_loss:
                best_loss = metric
                best_config = r.get("config")
        
        print(f"Found {len(existing_configs_map)} existing configurations with some runs completed.")

        stop_search = False
        
        # Track the last configuration index that was part of the original batch (or current HC batch)
        hc_batch_end_idx = len(combinations) - 1
        hc_step = 0

      
        def _run_hc_sync():
            nonlocal hc_step, hc_batch_end_idx
            self._wait_for_batch_completion(output_path, hc_batch_end_idx, n_runs)
            print(f"\\n--- Hill Climbing Step {hc_step + 1} Synchronization ---")
            self.load_existing_results(output_path)

            from collections import defaultdict
            config_scores = defaultdict(list)
            config_configs = {}
            for r in self.results:
                if r.get("status") == "success":
                    c_id = r.get("config_id")
                    val = r.get("test_rmse", r.get("val_loss", float('inf')))
                    config_scores[c_id].append(val)
                    if c_id not in config_configs:
                        config_configs[c_id] = r.get("config")

            current_best_avg_score = float('inf')
            current_best_config = None
            
            for c_id, scores in config_scores.items():
                if len(scores) > 0:
                    avg_score = sum(scores) / len(scores)
                    if avg_score < current_best_avg_score:
                        current_best_avg_score = avg_score
                        current_best_config = config_configs[c_id]

            if current_best_config is None:
                print("No successful robust configurations found yet. Skipping hill climbing.")
                return False

            hc_step += 1
            print(f"Evolving Best Robust Config (Avg RMSE: {current_best_avg_score:.6f}): {current_best_config}")

            neighbors = []
            for k, v in sorted(current_best_config.items()):
                if k in ["normalize", "solver", "filepath", "mask", "seed"]:
                    continue
                if isinstance(v, (int, float)):
                    is_log = any(log_key in k for log_key in ["learning_rate", "dt0", "rtol", "atol", "weight_decay"])
                    if is_log:
                        if v > 0:
                            for factor in [1.2, 1.5]:
                                n1 = current_best_config.copy(); n1[k] = v * factor; neighbors.append(n1)
                                n2 = current_best_config.copy(); n2[k] = v / factor; neighbors.append(n2)
                    else:
                        if isinstance(v, int):
                            for delta in [1, max(2, int(v * 0.15))]:
                                n1 = current_best_config.copy(); n1[k] = v + delta; neighbors.append(n1)
                                n2 = current_best_config.copy(); n2[k] = max(1, v - delta); neighbors.append(n2)
                        else:
                            for pct in [0.05, 0.15]:
                                delta = v * pct
                                n1 = current_best_config.copy(); n1[k] = v + delta; neighbors.append(n1)
                                n2 = current_best_config.copy(); n2[k] = v - delta; neighbors.append(n2)

            added_count = 0
            for n in neighbors:
                nt = tuple(sorted(n.items()))
                already_in_queue = any(tuple(sorted(c.items())) == nt for c in combinations)
                if not already_in_queue:
                    combinations.append(n)
                    added_count += 1

            if added_count > 0:
                hc_batch_end_idx = len(combinations) - 1
                print(f"Added {added_count} new neighbor configurations to explore in next batch.")
                return True
            else:
                print("No new neighbors found to explore. Hill climbing converged.")
                return False

        for config_idx, config in enumerate(combinations):
            if stop_search:
                break
            
            config_tuple = tuple(sorted(config.items()))
            existing_runs = existing_configs_map.get(config_tuple, [])
            
            if len(existing_runs) < n_runs:
                # Dynamically reload CSV silently to catch runs completed by parallel workers
                self.load_existing_results(output_path, silent=True)
                terminal_statuses = ["success", "failed", "recovered", "checkpoint_recovered"]
                existing_runs = [r["run_id"] for r in self.results if r.get("status") in terminal_statuses and tuple(sorted(r.get("config", {}).items())) == config_tuple]
                existing_configs_map[config_tuple] = existing_runs
            
            if len(existing_runs) >= n_runs:
                # Still check for Hill Climbing transition even if we skip
                if config_idx == hc_batch_end_idx and use_hill_climbing and hc_step < hill_climbing_steps:
                    _run_hc_sync()
                continue
            
            print(f"\\nProcessing Config {config_idx} (from total {len(combinations)}): {config}")
            config_dir = output_path / f"config_{config_idx}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            config_info_path = config_dir / "config_info.json"
            if not config_info_path.exists():
                with open(config_info_path, "w") as f:
                    json.dump({"config": config, "config_id": config_idx, "status": "running"}, f, indent=2)
            
            # Loop for multiple runs per configuration
            for run_idx in range(n_runs):
                if run_idx in existing_runs:
                    print(f"  Skipping existing Run {run_idx} for Config {config_idx}")
                    continue
                
                # Attempt atomic claim on the specific run
                run_dir = config_dir / f"run_{run_idx}"
                run_lock_path = output_path / f"run_{config_idx}_{run_idx}.lock"
                
               
                import os, time
                try:
                    fd = os.open(str(run_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                except FileExistsError:
                    # check if stale (older than 2 hours)
                    try:
                        if time.time() - run_lock_path.stat().st_mtime > 7200:
                            print(f"  Removing stale lock file: {run_lock_path.name}")
                            os.remove(run_lock_path)
                            fd = os.open(str(run_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                            os.close(fd)
                        else:
                            print(f"  Run {run_idx} of Config {config_idx} is currently locked by {run_lock_path.name}. Skipping.")
                            if config_idx == hc_batch_end_idx and use_hill_climbing and hc_step < hill_climbing_steps:
                                _run_hc_sync()
                            continue
                    except Exception as le: 
                        print(f"  Error handling lock file {run_lock_path.name}: {le}")
                        if config_idx == hc_batch_end_idx and use_hill_climbing and hc_step < hill_climbing_steps:
                            _run_hc_sync()
                        continue
                
                print(f"  Starting Run {run_idx}/{n_runs} for Config {config_idx}...")
                run_dir.mkdir(parents=True, exist_ok=True)
                
                losses, val_losses = None, None
                
                # Combine base params with current config
                current_init_params = self.base_params.copy()
                current_train_params = {
                    "val_split": val_split,
                    "max_steps": max_steps,
                    "patience": patience,
                    "print_freq": print_freq,
                    "timeout": timeout,
                    "nan_divergence_threshold": 20
                }
                
                # Separate config parameters into init params and train params
                train_keys = ["batch_size", "learning_rate", "lr", "weight_decay"]
                for k, v in config.items():
                    if k in train_keys:
                        if k == "learning_rate":
                            current_train_params["lr"] = v
                        else:
                            current_train_params[k] = v
                    else:
                        current_init_params[k] = v
                
                
                try:
                    n_samples = ys.shape[0]
                    indices = jnp.arange(n_samples)
                    
                    
                    run_seed = seed + run_idx
                    rng_np = np.random.default_rng(run_seed) 
                    perm = rng_np.permutation(np.array(indices))
                    
                    if n_train_total < n_samples:
                        n_test = n_samples - n_train_total
                    else:
                        n_test = 0
                        
                    test_idx = perm[:n_test]
                    train_idx = perm[n_test:]
                    
                    # Explicitly split training and validation data here
                    # This ensures run_meta.json records exactly what was used.
                    n_val_local = int(len(train_idx) * val_split)
                    if n_val_local > 0:
                        train_idx_actual = train_idx[:-n_val_local]
                        val_idx = train_idx[-n_val_local:]
                    else:
                        train_idx_actual = train_idx
                        val_idx = np.array([], dtype=int)

                    ys_train_actual = ys[train_idx_actual]
                    params_train_actual = params[train_idx_actual]
                    ys_val = ys[val_idx]
                    params_val = params[val_idx]

                    ts_test = ts
                    ys_test = ys[test_idx]
                    params_test = params[test_idx]

                    # Initialize model
                    model = self.model_class(current_init_params["filepath"], current_init_params["mask"], current_init_params)
                    
                    # Train model
                    start_time = time.time()
                    current_train_params["checkpoint_path"] = run_dir
                    current_train_params["checkpoint_freq"] = checkpoint_freq
                    current_train_params["seed"] = run_seed

                    losses, early_stopping_info, train_time, val_loss, val_losses, train_loss = model.train(
                        ts, 
                        ys_train_actual, 
                        params_train_actual, 
                        val_ys=ys_val, 
                        val_params=params_val, 
                        training_config=current_train_params
                    )
                    
                    stopping_condition = "completed"
                    nan_diverged = False
                    if early_stopping_info:
                        reason = early_stopping_info.get("reason", "")
                        stopping_condition = reason
                        if reason == "keyboard_interrupt":
                            stop_search = True
                            print("Global search interrupted by user inside training loop.")
                        elif reason in ("diverged", "first_step_timeout"):
                            raise RuntimeError(f"Training failed: {reason}")
                        elif reason == "nan_diverged":
                            nan_diverged = True
                            print(f"    Run {run_idx}: NaN divergence detected. Will attempt checkpoint recovery for predictions.")

                   
                    # For nan_diverged: skip using the final model (it has NaNs) and go straight to checkpoint recovery
                    recovered_cp_name = None
                    run_status = "success"

                    if nan_diverged:
                        # Find the last checkpoint without NaN predictions
                        test_preds = None
                        all_cps = sorted(list(run_dir.glob("step_*")), key=lambda x: int(x.name.split("_")[1]), reverse=True)
                        print(f"    Run {run_idx}: Found {len(all_cps)} checkpoints for nan_diverged recovery: {[cp.name for cp in all_cps]}")
                        for cp in all_cps:
                            try:
                                temp_model = self.model_class.load(cp)
                                cp_preds = []
                                for idx_t in range(len(ys_test)):
                                    p_t = params_test[idx_t]
                                    y0_t = ys_test[idx_t, 0, :]
                                    y_p = temp_model(ts_test, y0_t, p_t)
                                    cp_preds.append(y_p)
                                cp_preds = jnp.stack(cp_preds)
                                n_finite = int(jnp.sum(jnp.isfinite(cp_preds)))
                                n_total = int(cp_preds.size)
                                print(f"    Run {run_idx}: Checkpoint '{cp.name}' has {n_finite}/{n_total} finite prediction values.")
                                if jnp.any(jnp.isfinite(cp_preds)):
                                    test_preds = cp_preds
                                    recovered_cp_name = cp.name
                                    print(f"    Run {run_idx}: Using checkpoint '{cp.name}' for nan_diverged recovery predictions.")
                                    del temp_model
                                    break
                                del temp_model
                            except Exception as cp_load_err:
                                print(f"    Run {run_idx}: Checkpoint '{cp.name}' failed to load/predict: {cp_load_err}")
                                continue

                        if test_preds is not None:
                            # Save recovered predictions with a clear, distinct filename
                            jnp.save(run_dir / "test_preds_checkpoint_recovered.npy", test_preds)
                            print(f"    Run {run_idx}: Saved nan_diverged recovery predictions as 'test_preds_checkpoint_recovered.npy' (checkpoint: {recovered_cp_name}).")
                            run_status = "recovered"
                        else:
                            print(f"    Run {run_idx}: No clean checkpoint found for nan_diverged recovery. RMSE will be inf.")
                            test_preds = jnp.full_like(ys_test, float('nan'))
                            run_status = "recovered"  # still mark recovered (nan_diverged), not failed
                    else:
                        # Normal (non-diverged) path: use the final trained model
                        test_preds = []
                        for idx_t in range(len(ys_test)):
                            p_t = params_test[idx_t]
                            y0_t = ys_test[idx_t, 0, :]
                            y_p = model(ts_test, y0_t, p_t)
                            test_preds.append(y_p)
                        test_preds = jnp.stack(test_preds)
                        jnp.save(run_dir / "test_preds_final.npy", test_preds)

                    # Calculate RMSE
                    mask_t = jnp.isfinite(ys_test) & jnp.isfinite(test_preds)
                    if not jnp.any(mask_t):
                        rmse_val = float('inf')
                    else:
                        diff = ys_test - test_preds
                        safe_diff = jnp.where(mask_t, diff, 0.0)
                        rmse = jnp.sqrt(jnp.sum(jnp.square(safe_diff)) / jnp.sum(mask_t))
                        rmse_val = float(rmse)

                    print(f"    Run {run_idx} Test RMSE: {rmse_val:.6f}" + (f" (recovered from checkpoint: {recovered_cp_name})" if recovered_cp_name else ""))

                    # Save model
                    try:
                        model.save(str(run_dir))
                    except Exception as e:
                        print(f"Failed to save model to {run_dir}: {e}")

                    # Save losses over time
                    t_losses = [float(l) for l in losses]
                    v_losses = [float(l) for l in val_losses] if val_losses else []
                    
                    if len(v_losses) < len(t_losses):
                         v_losses += [None] * (len(t_losses) - len(v_losses))
                    elif len(v_losses) > len(t_losses):
                         v_losses = v_losses[:len(t_losses)]

                    history_df = pd.DataFrame({
                        "train_loss": t_losses,
                        "val_loss": v_losses
                    })
                    history_df.to_csv(run_dir / "loss_history.csv", index=False)
                    
                    # Save metadata
                    run_meta = {
                        "config_id": config_idx,
                        "run_id": run_idx,
                        "status": run_status,
                        "train_indices": [int(x) for x in train_idx_actual],
                        "validation_indices": [int(x) for x in val_idx],
                        "test_indices": [int(x) for x in test_idx],
                        "train_time_seconds": train_time,
                        "test_rmse": rmse_val,
                        "final_val_loss": float(val_loss),
                        "final_train_loss": float(train_loss),
                        "actual_steps_completed": len(losses),
                        "stopping_condition": stopping_condition,
                        "early_stopped": early_stopping_info is not None and early_stopping_info.get("stopped_early", False),
                        "config": config,
                        "recovered_checkpoint": recovered_cp_name,  # None unless nan_diverged recovery was used
                        "prediction_file": "test_preds_checkpoint_recovered.npy" if nan_diverged else "test_preds_final.npy"
                    }
                    with open(run_dir / "run_meta.json", "w") as f:
                        json.dump(run_meta, f, indent=2)

                    # Plotting per run
                    try:
                        plt.figure(figsize=(10, 5))
                        plt.plot(losses, label='Train Loss')
                        if val_losses:
                            plt.plot(val_losses, label='Validation Loss')
                        plt.xlabel('Step')
                        plt.ylabel('Loss')
                        plt.title(f'Run {run_idx} Dynamics (Config {config_idx})')
                        plt.yscale('log')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(run_dir / "loss_curve.png")
                        plt.close()
                    except Exception as e:
                        print(f"Plotting failed for run {run_idx}: {e}")

                    result = {
                        "config_id": config_idx,
                        "run_id": run_idx,
                        "config": config,
                        "val_loss": float(val_loss),
                        "test_rmse": rmse_val,
                        "training_time": train_time,
                        "iterations": len(losses),
                        "stopping_condition": stopping_condition,
                        "status": run_status,
                        "recovered_checkpoint": recovered_cp_name,
                        "prediction_file": "test_preds_checkpoint_recovered.npy" if nan_diverged else "test_preds_final.npy",
                        "output_dir": str(run_dir)
                    }
                    
                    if rmse_val < best_loss: 
                        best_loss = rmse_val
                        best_config = config
                        print(f"  New best RMSE: {best_loss:.6f}")
                    
                    self.results.append(result)
                    
                    # Cleanup resources to prevent slowdown over time
                    del model
                    gc.collect()
                    try:
                        jax.clear_caches()
                    except AttributeError:
                        # Fallback for older JAX versions
                        pass
                    plt.close('all')
                    
                except Exception as e:
                    print(f"Failed for config {config_idx} run {run_idx}: {e}")
                    import traceback
                    traceback.print_exc()

                  
                    # If the final model session failed (likely due to NaN), try to save predictions from the last good checkpoint
                    recovered_rmse = float('inf')
                    recovered_status = "failed"
                    recovered_cp_name = None
                    
                    try:
                        all_cps = sorted(list(run_dir.glob("step_*")), key=lambda x: int(x.name.split("_")[1]), reverse=True)
                        if all_cps:
                            for latest_cp in all_cps[:3]:
                                try:
                                    temp_model = self.model_class.load(latest_cp)
                                    cp_test_preds = []
                                    for idx_t in range(len(ys_test)):
                                        p_t = params_test[idx_t]
                                        y0_t = ys_test[idx_t, 0, :]
                                        y_p = temp_model(ts_test, y0_t, p_t)
                                        cp_test_preds.append(y_p)
                                    cp_test_preds = jnp.stack(cp_test_preds)
                                    jnp.save(run_dir / "test_preds_checkpoint.npy", cp_test_preds)
                                    
                                    # Calculate recovered RMSE
                                    mask_cp = jnp.isfinite(ys_test) & jnp.isfinite(cp_test_preds)
                                    if jnp.any(mask_cp):
                                        diff_cp = ys_test - cp_test_preds
                                        safe_diff_cp = jnp.where(mask_cp, diff_cp, 0.0)
                                        recovered_rmse = float(jnp.sqrt(jnp.sum(jnp.square(safe_diff_cp)) / jnp.sum(mask_cp)))
                                        recovered_status = "recovered"
                                        recovered_cp_name = latest_cp.name
                                        print(f"    Recovered RMSE {recovered_rmse:.6f} from {latest_cp.name}")
                                    
                                    del temp_model
                                    break
                                except Exception:
                                    continue
                    except Exception as cp_err:
                        print(f"  Could not save checkpoint predictions: {cp_err}")
                    
                    result = {
                        "config_id": config_idx,
                        "run_id": run_idx,
                        "config": config,
                        "error": str(e),
                        "status": recovered_status,
                        "val_loss": float('inf'),
                        "test_rmse": recovered_rmse,
                        "model_source": recovered_cp_name if recovered_cp_name else "failed"
                    }
                    self.results.append(result)
                    
                    # Save failure metadata so resume logic skips this run if we recovered
                    try:
                        run_dir.mkdir(parents=True, exist_ok=True)
                        failed_meta = {
                            "config_id": config_idx,
                            "run_id": run_idx,
                            "status": recovered_status,
                            "error": str(e),
                            "test_rmse": recovered_rmse,
                            "config": config,
                            "test_indices": [int(x) for x in test_idx],
                            "model_source": recovered_cp_name if recovered_cp_name else "failed"
                        }
                        with open(run_dir / "run_meta.json", "w") as f:
                            json.dump(failed_meta, f, indent=2)
                    except Exception as meta_err:
                        print(f"  Could not save failure metadata: {meta_err}")
                        
                    # Save loss history for failed runs if they were partially generated
                    if losses is not None and len(losses) > 0:
                        try:
                            fail_val_losses = val_losses if val_losses is not None else []
                            if len(fail_val_losses) < len(losses):
                                fail_val_losses += [None] * (len(losses) - len(fail_val_losses))
                            elif len(fail_val_losses) > len(losses):
                                fail_val_losses = fail_val_losses[:len(losses)]
                            pd.DataFrame({
                                "train_loss": [float(l) for l in losses],
                                "val_loss": [float(l) if l is not None else None for l in fail_val_losses]
                            }).to_csv(run_dir / "loss_history.csv", index=False)
                            plt.figure(figsize=(10, 5))
                            plt.plot(losses, label='Train Loss')
                            if any(l is not None for l in fail_val_losses):
                                plt.plot(fail_val_losses, label='Validation Loss')
                            plt.xlabel('Step')
                            plt.ylabel('Loss')
                            plt.title(f'Failed Run {run_idx} Dynamics (Config {config_idx})')
                            plt.yscale('log')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plt.savefig(run_dir / "loss_curve.png")
                            plt.close()
                        except Exception as p_err:
                            print(f"  Failed saving loss curve for failed run: {p_err}")

                    # Cleanup resources even on failure
                    try:
                        del model
                    except NameError:
                        pass
                    gc.collect()
                    try:
                        jax.clear_caches()
                    except AttributeError:
                        pass
                    plt.close('all')
                
                self.save_results(output_path)
    
      
            # Calculate Standard Deviation and Stats after all runs for this config
            config_results = [r for r in self.results if r.get("config_id") == config_idx and r.get("status") == "success"]
            if len(config_results) > 1:
                rmses = [r["test_rmse"] for r in config_results]
                val_losses_config = [r["val_loss"] for r in config_results]
                times = [r["training_time"] for r in config_results]
                
                stats = {
                    "config_id": config_idx,
                    "n_runs": len(config_results),
                    "mean_test_rmse": float(np.mean(rmses)),
                    "std_test_rmse": float(np.std(rmses)),
                    "median_test_rmse": float(np.median(rmses)),
                    "max_test_rmse": float(np.max(rmses)),
                    "min_test_rmse": float(np.min(rmses)),
                    "robust_score": float(np.mean(rmses) + np.std(rmses)), # Penalize instability
                    "mean_val_loss": float(np.mean(val_losses_config)),
                    "std_val_loss": float(np.std(val_losses_config)),
                    "mean_training_time": float(np.mean(times)),
                    "std_training_time": float(np.std(times)),
                    "config": config
                }
                
                with open(config_dir / "config_summary.json", "w") as f:
                    json.dump(stats, f, indent=2)
                
                # Mark as completed in config_info.json
                try:
                    with open(config_info_path, "r") as f:
                        info = json.load(f)
                    info["status"] = "completed"
                    info["finished_at"] = time.time()
                    with open(config_info_path, "w") as f:
                        json.dump(info, f, indent=2)
                except Exception as e:
                    print(f"  Warning: Could not update status in config_info.json: {e}")

                print(f"  Config {config_idx} Stats -> Mean RMSE: {stats['mean_test_rmse']:.6f} (±{stats['std_test_rmse']:.6f} | Robust: {stats['robust_score']:.6f})")
                
                # Config Visualizations
                try:
                    # 1. Combined Loss Curves
                    plt.figure(figsize=(10, 6))
                    for r_idx in range(n_runs):
                        hist_file = config_dir / f"run_{r_idx}" / "loss_history.csv"
                        if hist_file.exists():
                            df_h = pd.read_csv(hist_file)
                            plt.plot(df_h["train_loss"], alpha=0.3, label=f'Run {r_idx}' if n_runs <= 5 else None)
                    
                    plt.yscale('log')
                    plt.title(f'Config {config_idx} Aggregate Training Loss')
                    plt.xlabel('Step')
                    plt.ylabel('Loss')
                    if n_runs <= 5: plt.legend()
                    plt.savefig(config_dir / "aggregate_loss.png")
                    plt.close()
                    
                    # 2. Boxplot of RMSEs
                    plt.figure(figsize=(6, 4))
                    plt.boxplot(rmses)
                    plt.title(f'RMSE Distribution (Config {config_idx})')
                    plt.ylabel('Test RMSE')
                    plt.xticks([1], ['Runs'])
                    plt.savefig(config_dir / "rmse_distribution.png")
                    plt.close()
                except Exception as e:
                    print(f"Config plotting failed: {e}")

         
            # When we reach the end of the current batch, run the HC sync via the helper.
            if config_idx == hc_batch_end_idx and use_hill_climbing and hc_step < hill_climbing_steps:
                _run_hc_sync()

        print(f"\nTuning complete. Best overall Test RMSE: {best_loss:.6f}")
        print(f"Best config: {best_config}")
        
        return self.save_results(output_path)
