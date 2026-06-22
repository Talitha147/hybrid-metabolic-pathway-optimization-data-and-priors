import os
import sys
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import fcntl
import time
import shutil
import jax
import jax.numpy as jnp

root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from scripts.models.hybrid_model import HybridModel
from scripts.models.NODE import NODEModel
from scripts.ExperimentRunner import ExperimentRunner
import concurrent.futures


def _get_ts_indices_from_name(name):
   
    m = re.search(r'_(\d+)_points', name) or re.search(r'_(\d+)_steps', name)
    if not m:
        return None
    n = int(m.group(1))
    return list(map(int, np.linspace(0, 39, n + 1).astype(int)))


def _find_tracker(run_dir, stop_at):
 
    current = run_dir.parent
    while True:
        candidate = current / "experiment_tracker.csv"
        if candidate.exists():
            return candidate
        if current == stop_at or current == stop_at.parent:
            break
        current = current.parent
    return None

def get_best_step(val_losses, print_freq, min_step, window_size, available_steps, tolerance=0.05):
 
    if len(val_losses) == 0:
        return None

    final_step = max(available_steps)

    steps = np.arange(len(val_losses)) * print_freq

    if len(val_losses) >= window_size:
        smoothed = np.convolve(val_losses, np.ones(window_size) / window_size, mode='valid')
        smoothed_steps = steps[window_size - 1:]
    else:
        smoothed = val_losses
        smoothed_steps = steps

    def _smoothed_score(step):
        idx = min(range(len(smoothed_steps)), key=lambda i: abs(smoothed_steps[i] - step))
        return float(smoothed[idx])

    # Filter available checkpoints that meet min_step
    valid_checkpoints = sorted([s for s in available_steps if s >= min_step])
    if not valid_checkpoints:
        return None

    final_score = _smoothed_score(final_step)
   
    improvement_threshold = final_score * (1.0 - tolerance)

    # Find the global minimum among valid checkpoints
    best_score = float('inf')
    best_step_found = final_step
    
    for s in valid_checkpoints:
        score = _smoothed_score(s)
        if score <= best_score: 
            best_score = score
            best_step_found = s

    # If the best found step isn't significantly better than the final step, keep final
    if best_score >= improvement_threshold:
         return final_step
        
    print(f"  Selection: picked step {best_step_found} (score {best_score:.8f}) over final (threshold {improvement_threshold:.8f})")
    return best_step_found

def log_refinement_to_csv(log_path, entry):
    
    log_path = Path(log_path)
    df = pd.DataFrame([entry])
    
    lock_path = log_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if log_path.exists():
                df.to_csv(log_path, mode='a', header=False, index=False)
            else:
                df.to_csv(log_path, mode='w', header=True, index=False)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

def reevaluate_run(run_dir, best_step, model_class, runner, tracker_path=None, log_path=None):

    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    final_step = meta.get("actual_steps_completed", 0)
    is_final = (best_step == final_step)
    original_rmse = meta.get("test_rmse_all_ts_original", 
                             meta.get("test_rmse_original", 
                                      meta.get("test_rmse_all_ts", meta.get("test_rmse"))))
    
    current_rmse = meta.get("test_rmse_all_ts", meta.get("test_rmse"))

    log_entry = {
        "experiment": run_dir.parents[1].name,
        "subset": run_dir.parent.name,
        "seed": run_dir.name,
        "old_rmse": original_rmse,
        "new_rmse": None,
        "checkpoint_used": f"step_{best_step}" if not is_final else "final",
        "steps": final_step,
        "time": meta.get("train_time_seconds"),
        "is_final_model": is_final,
        "recovered": meta.get("status_original", meta.get("status")) == "recovered"
    }

    if is_final and meta.get("refined_checkpoint") in ["final", f"step_{final_step}"]:
        log_entry["new_rmse"] = original_rmse
        if log_path: log_refinement_to_csv(log_path, log_entry)
        return True, "final"

    cp_dir = run_dir / "checkpoints" / f"step_{best_step}"
    
    if not cp_dir.exists() and is_final:
        cp_dir = run_dir
        
    if not cp_dir.exists():
        return False, None
        

    for f in ["model.eqx", "results.json", "predictions.npz"]:
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, run_dir / f"{f}.backup")
            
    for key in ["test_rmse", "test_rmse_all_ts", "test_rmse_train_ts", "status"]:
        if key in meta and f"{key}_original" not in meta:
            meta[f"{key}_original"] = meta[key]

    try:
        best_model = model_class.load(str(cp_dir))
    except Exception:
        return False, None
        
    run_meta = meta
    test_idx = run_meta["test_indices"]
    train_idx = [i for i in run_meta.get("train_indices", []) if i not in run_meta.get("validation_indices", [])]

    test_res = runner.get_preds(best_model, test_idx)
    
    # Extract metrics
    rmse_all = test_res["RMSE_All_Species_All_TS"]
    
    # Only apply the checkpoint if it actually improves the test RMSE
    # relative to the model state we currently have in the folder.
    if rmse_all >= current_rmse:

        current_cp = meta.get("refined_checkpoint", "final")
        print(f"  Proposed step {best_step} ({rmse_all:.6f}) is not better than current model {current_cp} ({current_rmse:.6f}). Keeping current.")
        
        meta["is_refined"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        log_entry["new_rmse"] = current_rmse # The RMSE of the model that stays in the folder
        log_entry["checkpoint_used"] = current_cp
        log_entry["is_final_model"] = (current_cp == "final" or current_cp == f"step_{final_step}")
        if log_path: log_refinement_to_csv(log_path, log_entry)
        return True, current_cp

    train_res = runner.get_preds(best_model, train_idx)
    rmse_train_ts = test_res["RMSE_All_Species_Train_TS"]
    
    # Update run_meta.json
    run_meta["test_rmse"] = float(rmse_all)
    run_meta["test_rmse_all_ts"] = float(rmse_all)
    run_meta["test_rmse_train_ts"] = float(rmse_train_ts)
    
    # Add exclusion info to run_meta
    exc = test_res.get("ExclusionInfo", {})
    run_meta["eval_total_points"] = exc.get("eval_total_points", exc.get("test_total_points"))
    run_meta["eval_excluded_points"] = exc.get("eval_excluded_points", exc.get("test_excluded_points"))
    run_meta["eval_excluded_fraction"] = exc.get("eval_excluded_fraction", exc.get("test_excluded_fraction"))
    
    run_meta["status"] = "success"
    run_meta["refined_checkpoint"] = f"step_{best_step}"
    run_meta["is_refined"] = True
    
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
        
    best_model.save(str(run_dir))
    
    # Regenerate results.json 
    results = {
        "ModelType": str(model_class.__name__),
        "RunID": run_dir.name,
        "RMSE_All_Species_All_TS": test_res["RMSE_All_Species_All_TS"],
        "RMSE_All_Species_Train_TS": test_res["RMSE_All_Species_Train_TS"],
        "RMSE_pCA_Final": test_res["RMSE_pCA_Final"],
        "NRMSE_pCA_Final": test_res["NRMSE_pCA_Final"],
        "RMSE_pCA_All": test_res["RMSE_pCA_All"],
        "NRMSE_All_Species": test_res["NRMSE_All_Species"],
        "final_pCA_preds": test_res["final_pCA_preds"],
        "TestIndices": test_res["Indices"],
        "NRMSE_per_species": test_res["NRMSE_per_species"],
        "rmse_per_strain": test_res["rmse_per_strain"],
        "nrmse_per_strain": test_res["nrmse_per_strain"],
        "ts_indices": runner.ts_indices.tolist(),
        "ExclusionInfo": test_res["ExclusionInfo"],

        "Train_RMSE_All_Species_All_TS": train_res["RMSE_All_Species_All_TS"],
        "Train_RMSE_All_Species_Train_TS": train_res["RMSE_All_Species_Train_TS"],
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
        
    np.savez_compressed(
        run_dir / "predictions.npz", 
        Predictions=test_res["Predictions"], 
        Train_Predictions=train_res["Predictions"]
    )
    
    gt_path = run_dir.parent / "ground_truths.npz"
    if not gt_path.exists():
        np.savez_compressed(
            gt_path,
            GroundTruths=test_res["GroundTruths"],
            Train_GroundTruths=train_res["GroundTruths"]
        )
    
        
    log_entry["new_rmse"] = rmse_all
    if log_path: log_refinement_to_csv(log_path, log_entry)
    
    checkpoint_str = f"step_{best_step}"
    print(f"  Successfully refined to {checkpoint_str} | New RMSE: {rmse_all:.6f}")
    return True, checkpoint_str



def generate_report(experiments_base, pattern, report_path, min_step, window_size):
    """Scans all runs and generates a CSV report with current and proposed checkpoints."""
    print(f"Scanning for runs in {experiments_base}...")
    all_run_metas = list(experiments_base.rglob("run_meta.json"))
    print(f"Found {len(all_run_metas)} runs.")
    report_entries = []
    for meta_path in all_run_metas:
        seed_dir = meta_path.parent
        
        # Filter by pattern if provided
        if pattern != "pCA_*" and pattern not in str(seed_dir):
            continue
            
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"  Warning: Corrupted or missing run_meta.json in {seed_dir}. Skipping.")
            continue
        
        model_type = meta.get("model_type")
        val_history_path = seed_dir / "val_loss_history.npy"
        
        proposed_step = None
        if val_history_path.exists() and model_type in ["HybridModel", "NODEModel"]:
            val_losses = np.load(val_history_path)
            cp_base = seed_dir / "checkpoints"
            
            available_steps = []
            if cp_base.exists():
                for cp_dir in cp_base.glob("step_*"):
                    try:
                        if cp_dir.is_dir() and (cp_dir / "model.eqx").exists():
                            step = int(cp_dir.name.split("_")[1])
                            available_steps.append(step)
                    except: pass
            
            # Incorporate the final state as a candidate for selection
            final_step = meta.get("actual_steps_completed", 0)
            if final_step > 0:
                available_steps.append(final_step)
            available_steps = sorted(list(set(available_steps)))
            
            if available_steps:
                print_freq = meta.get("print_freq", 50)
                proposed_step = get_best_step(val_losses, print_freq, min_step, window_size, available_steps)
    
        entry = {
            "Experiment": seed_dir.parents[1].name,
            "Subset": seed_dir.parent.name,
            "Seed": seed_dir.name,
            "ModelType": model_type,
            "Status": meta.get("status"),
            "FinalStep": meta.get("actual_steps_completed"),
            "UsedCheckpoint": meta.get("refined_checkpoint", "final"),
            "ProposedStep": proposed_step if proposed_step is not None else "N/A",
            "RMSE": meta.get("test_rmse"),
            "OriginalStatus": meta.get("original_status", meta.get("status"))
        }
        report_entries.append(entry)
    if report_entries:
        report_df = pd.DataFrame(report_entries)
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path, index=False)
        print(f"Report saved to {report_path}")
    else:
        print("No runs found matching the criteria.")

def reconstruct_log(experiments_base, pattern, log_path):
    """Reconstructs refinement_log.csv from metadata files."""
    print(f"Reconstructing refinement log in {experiments_base}...")
    all_run_metas = list(experiments_base.rglob("run_meta.json"))
    log_entries = []
    
    for meta_path in all_run_metas:
        seed_dir = meta_path.parent
        if pattern != "pCA_*" and pattern not in str(seed_dir):
            continue
            
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except: continue
        
        if not meta.get("is_refined"):
            continue
            
        final_step = meta.get("actual_steps_completed", 0)
        refined_cp = meta.get("refined_checkpoint", "final")
        is_final = (refined_cp == f"step_{final_step}" or refined_cp == "final")
        
        entry = {
            "experiment": seed_dir.parents[1].name,
            "subset": seed_dir.parent.name,
            "seed": seed_dir.name,
            "old_rmse": meta.get("test_rmse_all_ts_original", meta.get("test_rmse_original")),
            "new_rmse": meta.get("test_rmse_all_ts", meta.get("test_rmse")),
            "checkpoint_used": refined_cp if not is_final else "final",
            "steps": final_step,
            "time": meta.get("train_time_seconds"),
            "is_final_model": is_final,
            "recovered": meta.get("status_original", meta.get("status")) == "recovered"
        }
        log_entries.append(entry)
        
    if log_entries:
        log_df = pd.DataFrame(log_entries)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(log_path, index=False)
        print(f"Reconstructed log saved to {log_path} ({len(log_entries)} entries)")
    else:
        print("No refined runs found to reconstruct log.")

_global_data = None

def init_worker(model_path, data_tuple, mask):

    global _global_data
    _global_data = (model_path, data_tuple, mask)

def _make_runner(model_class, ts_indices):
 
    model_path, data_tuple, mask = _global_data
    return ExperimentRunner(
        model_path, ts_indices, data_tuple,
        mask=mask.tolist() if hasattr(mask, 'tolist') else list(mask),
        model_class=model_class
    )

def process_single_run(meta_path, experiments_base, args):
    
    seed_dir = meta_path.parent
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except:
        return None

    if meta.get("is_refined") and not args.force:
        return f"Skipped: {seed_dir.relative_to(experiments_base)} (already refined)"

    model_type = meta.get("model_type")
    if model_type not in ["HybridModel", "NODEModel"]:
        return None

    val_history_path = seed_dir / "val_loss_history.npy"
    if not val_history_path.exists():
        return None

    val_losses = np.load(val_history_path)
    cp_base = seed_dir / "checkpoints"

    available_steps = []
    if cp_base.exists():
        for cp_dir in cp_base.glob("step_*"):
            try:
                if cp_dir.is_dir() and (cp_dir / "model.eqx").exists():
                    available_steps.append(int(cp_dir.name.split("_")[1]))
            except:
                pass

    final_step = meta.get("actual_steps_completed", 0)
    if final_step > 0:
        available_steps.append(final_step)
    available_steps = sorted(list(set(available_steps)))

    if not available_steps:
        return None

    print_freq = meta.get("print_freq", 50)

    # Determine ts_indices for this run
    ts_indices = meta.get("ts_indices")
    if ts_indices is None:
        ts_indices = _get_ts_indices_from_name(seed_dir.parents[1].name)

    n_ts = len(ts_indices) if ts_indices else 0
    if n_ts <= 2:  # t=0 + 1 measurement point
        best_step = final_step
    else:
        adaptive_min_step = max(args.min_step, int(final_step * 0.20))
        best_step = get_best_step(val_losses, print_freq, adaptive_min_step,
                                  args.window_size, available_steps, args.tolerance)
    if best_step is None:
        return None

    if args.dry_run:
        checkpoint_str = "final" if best_step == final_step else f"step_{best_step}"
        return f"Dry-run: {seed_dir.relative_to(experiments_base)} -> {checkpoint_str}"

    current_model_class = HybridModel if model_type == "HybridModel" else NODEModel

    if ts_indices is None:
        return f"Failed (no ts_indices): {seed_dir.relative_to(experiments_base)}"

    current_runner = _make_runner(current_model_class, ts_indices)

    tracker = args.tracker or None
    if not tracker:
        found = _find_tracker(seed_dir, experiments_base)
        if found:
            tracker = str(found)

    success, checkpoint_str = reevaluate_run(seed_dir, best_step, current_model_class, current_runner,
                             tracker_path=tracker, log_path=args.log_path)
    if success:
        return f"Refined: {seed_dir.relative_to(experiments_base)} -> {checkpoint_str}"
    return f"Failed: {seed_dir.relative_to(experiments_base)}"

def main():
  
    # experiments_dir = "Experiments_actual/Question_1"
    experiments_dir = "Experiments_actual/Question_2"

    # Minimum step to consider for checkpoint selection
    min_step = 200

    # Window size for validation loss smoothing
    window_size = 3

    # Tolerance for picking final step over best checkpoint
    tolerance = 0.005
    
    # Pattern for experiment directories to match
    pattern = "pCA_*"
    # pattern = "pCA_hybrid_1_points*"
    
    # Force re-evaluation even if already refined
    force = False
    # force = True

    # Just print what would be done
    dry_run = False
    # dry_run = True

    # Just generate a CSV report of used checkpoints
    report_only = False
    # report_only = True

    # Reconstruct refinement_log.csv from metadata
    reconstruct_log = False
    # reconstruct_log = True

    # Path to central tracker CSV to update
    tracker = None
    
    # Number of parallel workers
    workers = 4

    report_path = os.path.join(experiments_dir, "checkpoint_usage_report.csv")
    log_path = os.path.join(experiments_dir, "refinement_log.csv")


    from types import SimpleNamespace
    args = SimpleNamespace(
        experiments_dir=experiments_dir,
        min_step=min_step,
        window_size=window_size,
        tolerance=tolerance,
        tracker=tracker,
        force=force,
        pattern=pattern,
        dry_run=dry_run,
        report_only=report_only,
        report_path=report_path,
        log_path=log_path,
        reconstruct_log=reconstruct_log,
        workers=workers
    )
    
    experiments_base = Path(args.experiments_dir)
    
    if args.reconstruct_log:
        reconstruct_log(experiments_base, args.pattern, args.log_path)
        return

    if args.report_only:
        generate_report(experiments_base, args.pattern, args.report_path, args.min_step, args.window_size)
        return

    # Initialize runners once
    model_path = 'models/pCA_model_changed_S.xml'
    from scripts.data_generation import load_data_from_csvs
    ts, ys_total, params_total, _ = load_data_from_csvs(csv_dir="data", metabolites=['substrate', 'c_biomass', 'co2', 'e4p', 'pep', 'dhap', 'epsp', 'pcoumaric_acid'], model_name="pCA_model")
    data_tuple = (jnp.arange(40), ys_total, params_total)
    
    M = jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    
    all_run_metas = list(experiments_base.rglob("run_meta.json"))
    filtered_metas = [m for m in all_run_metas if (args.pattern == "pCA_*" or args.pattern in str(m))]
    
    print(f"Starting refinement for {len(filtered_metas)} candidate runs using {args.workers} workers...")
    
    results_counter = {"success": 0, "skipped": 0, "failed": 0, "dry": 0}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(model_path, data_tuple, M)
    ) as executor:
        futures = {executor.submit(process_single_run, meta_path, experiments_base, args): meta_path for meta_path in filtered_metas}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                if result.startswith("Refined"): results_counter["success"] += 1
                elif result.startswith("Skipped"): results_counter["skipped"] += 1
                elif result.startswith("Failed"): results_counter["failed"] += 1
                elif result.startswith("Dry-run"): results_counter["dry"] += 1
                
                if i % 10 == 0 or i == len(filtered_metas) - 1:
                    print(f"[{i+1}/{len(filtered_metas)}] {result}")
            
    print(f"\nRefinement Complete!")
    print(f"Summary: {results_counter['success']} Refined, {results_counter['skipped']} Skipped, {results_counter['failed']} Failed, {results_counter['dry']} Dry-run")

    # generate_report(experiments_base, args.pattern, args.report_path, args.min_step, args.window_size)



if __name__ == "__main__":
    main()