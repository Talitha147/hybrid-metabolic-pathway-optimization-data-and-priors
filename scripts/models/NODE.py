import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import diffrax
import optax
import pandas as pd
import numpy as np
import time
from pathlib import Path
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from typing import List


class NODEModel:
    model: "NODE_Integrator"

    def __init__(self, filepath, mask: List[int] = None, model_config: dict = None):
        if model_config is None:
            model_config = {}
            
        self.seed = model_config.get("seed", 0)
        self.key = jr.PRNGKey(self.seed)
        self.filepath = filepath
        
        self.sbmlModel = SBMLModel(self.filepath)

        self.n_metabolites = self.sbmlModel.y0.shape[0]
        self.n_params = len(self.sbmlModel.parameters.items())

        param_names = tuple(self.sbmlModel.parameters.keys())
        self.param_names = param_names
        self.param_index = {k: i for i, k in enumerate(param_names)}
        
        self.normalized = model_config.get("normalize", True)
        self.dropout = model_config.get("dropout", 0.1)
        self.width_NODE = model_config.get("width_NODE", 32)
        self.depth_NODE = model_config.get("depth_NODE", 2)
        self.solver = model_config.get("solver", "Kvaerno5")
        self.dt0 = model_config.get("dt0", 1e-4)
        self.rtol = model_config.get("rtol", 1e-4)
        self.atol = model_config.get("atol", 1e-6)

        self.model = NODE_Integrator(state_dim=self.n_metabolites, 
                                     param_dim=self.n_params, 
                                     width=self.width_NODE, 
                                     depth=self.depth_NODE, 
                                     key=self.key,
                                     dropout = self.dropout,
                                     normalized = self.normalized,
                                     solver=self.solver,
                                     dt0=self.dt0, 
                                     rtol=self.rtol, 
                                     atol=self.atol)
    @staticmethod
    @eqx.filter_value_and_grad
    def loss_fn(model, ts, batch_y0, batch_params, batch_ys, key, weight_decay):
     
        batch_size = batch_y0.shape[0]
        keys = jr.split(key, batch_size)

        def rollout(y0, params, k):
            return model(ts, y0, params, key=k)
        
        y_pred = jax.vmap(rollout)(batch_y0, batch_params, keys)

        # Mask out inf/nan values which can occur during training
        mask = jnp.isfinite(y_pred) & jnp.isfinite(batch_ys)
        diff = y_pred - batch_ys
        safe_diff = jnp.where(mask, diff, 0.0) 
        mse_loss = jnp.mean(jnp.square(safe_diff)) 
        
        # Add penalty for NaNs to guide optimizer away from unstable regions
        nan_count = jnp.sum(~mask)
        nan_penalty = 1e9 * (nan_count / batch_ys.size)
        
        nn_params = eqx.filter(model.func.nn_flux, eqx.is_array)
        l2_loss = weight_decay * jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(nn_params)]))

        total_loss = mse_loss + nan_penalty + l2_loss

        return jnp.nan_to_num(total_loss, nan=1e9, posinf=1e9, neginf=1e9)

    @staticmethod
    @eqx.filter_jit
    def train_step(model, opt_state, ts, batch_y0, batch_params, batch_ys, optimizer, key, weight_decay):
       
        loss, grads = NODEModel.loss_fn(model, ts, batch_y0, batch_params, batch_ys, key, weight_decay)
        
        updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
            )
        
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @staticmethod
    def dataloader(y0s, params, ys, batch_size, *, key):
        n = ys.shape[0]
        indices = jnp.arange(n)

        while True:
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, indices)

            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                yield y0s[idx], params[idx], ys[idx]

    def train(self, ts, ys, params, val_ys=None, val_params=None, training_config: dict = None):
        if training_config is None:
            training_config = {}
            
        val_split = training_config.get("val_split", 0.2)
        n_val_strains = training_config.get("n_val_strains", 0)
        max_steps = training_config.get("max_steps", 2000)
        batch_size = training_config.get("batch_size", 8)
        patience = training_config.get("patience", 50)
        min_delta = training_config.get("min_delta", 1e-5)
        checkpoint_path = training_config.get("checkpoint_path", None)
        lr = training_config.get("learning_rate", training_config.get("lr", 1e-3))
        weight_decay = training_config.get("weight_decay", 1e-5)
        checkpoint_freq = training_config.get("checkpoint_freq", 100)
        resume = training_config.get("resume", False)
        print_freq = training_config.get("print_freq", 100)
        step_timeout_threshold = training_config.get("step_timeout_threshold", 30)
        timeout = training_config.get("timeout", None)
        # Divergence / first-step checks
        divergence_loss_threshold = training_config.get("divergence_loss_threshold", 1e7)
        first_step_timeout = training_config.get("first_step_timeout", 120)
        lr_warmup_steps = training_config.get("lr_warmup_steps", 100)
    
        # Split training and validation
        n = ys.shape[0]
        if n_val_strains > 0:
            n_val = n_val_strains
        else:
            n_val = int(n * val_split)
        losses = []
        val_losses = []
        
        training_start_time = time.time()
        
        if val_ys is not None:
            # Use explicitly provided validation set
            train_y0s, train_params, train_ys = ys[:, 0, :], params[:], ys[:]
            val_y0s, val_params, val_ys = val_ys[:, 0, :], val_params[:], val_ys[:]
            n_val = val_ys.shape[0]
            print(f"Training with {train_ys.shape[0]} samples, validating with {n_val} samples (explicitly provided)")
        elif n_val > 0:
            train_y0s, train_params, train_ys = ys[:-n_val, 0, :], params[:-n_val], ys[:-n_val]
            val_y0s, val_params, val_ys = ys[-n_val:, 0, :], params[-n_val:], ys[-n_val:]
            print(f"Training with {n-n_val} samples, validating with {n_val} samples")
        else:
            train_y0s, train_params, train_ys = ys[:, 0, :], params[:], ys[:]
            val_y0s, val_params, val_ys = ys[:0, 0, :], params[:0], ys[:0]
            print(f"Training with {n} samples, no validation set")

    
        # Necessary for data normalization
        if self.normalized:
            
            if train_ys.size > 0:
                y_min = jnp.min(train_ys, axis=(0, 1))
                y_max = jnp.max(train_ys, axis=(0, 1))
                y_range = jnp.where((y_max - y_min) < 1e-4, 1.0, y_max - y_min)
                
                p_min = jnp.min(train_params, axis=0)
                p_max = jnp.max(train_params, axis=0)
                p_range = jnp.where((p_max - p_min) < 1e-4, 1.0, p_max - p_min)
            else:
                # Fallback for empty data
                y_min = jnp.zeros(ys.shape[2])
                y_range = jnp.ones(ys.shape[2])
                p_min = jnp.zeros(params.shape[1])
                p_range = jnp.ones(params.shape[1])
            
            
            # Static fields cannot be updated with eqx.tree_at; reconstruct the
            # model pytree by swapping the inner NODE_Func for a new one that
            # carries the data-derived normalisation constants as static values.
            old_func = self.model.func
            new_func = NODE_Func.__new__(NODE_Func)
            object.__setattr__(new_func, "nn_flux",    old_func.nn_flux)
            # Static fields must be plain tuples (not arrays) to be
            # hashable/comparable for JIT caching purposes.
            object.__setattr__(new_func, "y_min",      tuple(jax.device_get(y_min).tolist()))
            object.__setattr__(new_func, "y_range",    tuple(jax.device_get(y_range).tolist()))
            object.__setattr__(new_func, "p_min",      tuple(jax.device_get(p_min).tolist()))
            object.__setattr__(new_func, "p_range",    tuple(jax.device_get(p_range).tolist()))
            object.__setattr__(new_func, "normalized", True)
            object.__setattr__(new_func, "nn_scale",   old_func.nn_scale)
            self.model = eqx.tree_at(lambda m: m.func, self.model, new_func)
            
            # Normalize Data
            train_y0s = (train_y0s - y_min) / y_range
            train_ys = (train_ys - y_min) / y_range
            train_params = (train_params - p_min) / p_range
            
            if n_val > 0:
                val_y0s = (val_y0s - y_min) / y_range
                val_ys = (val_ys - y_min) / y_range
                val_params = (val_params - p_min) / p_range
    

        # LR warmup prevents the first large gradient update from destroying the ODE.
        warmup_steps = min(lr_warmup_steps, max_steps // 4)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=max_steps,
            end_value=lr * 0.1,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adabelief(lr_schedule),
        )

        opt_state = optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

        start_step = 0
        best_val_loss = float('inf')
        best_train_loss = float('inf')
     
        # Resume training from checkpoint
        if resume and checkpoint_path and Path(checkpoint_path).exists():
            try:
                import json
                path = Path(checkpoint_path)
                with open(path / "metadata.json", "r") as f:
                    metadata = json.load(f)
                
                self.model = eqx.tree_deserialise_leaves(path / "model.eqx", self.model)
                
                if (path / "opt_state.eqx").exists():
                     opt_state = eqx.tree_deserialise_leaves(path / "opt_state.eqx", opt_state)
                
                if self.normalized and "y_min" in metadata and metadata["y_min"] is not None:
                    old_func = self.model.func
                    new_func = NODE_Func.__new__(NODE_Func)
                    object.__setattr__(new_func, "nn_flux",    old_func.nn_flux)
                    object.__setattr__(new_func, "y_min",      tuple(metadata["y_min"]))
                    object.__setattr__(new_func, "y_range",    tuple(metadata["y_range"]))
                    object.__setattr__(new_func, "p_min",      tuple(metadata["p_min"]))
                    object.__setattr__(new_func, "p_range",    tuple(metadata["p_range"]))
                    object.__setattr__(new_func, "normalized", True)
                    object.__setattr__(new_func, "nn_scale",   old_func.nn_scale)
                    self.model = eqx.tree_at(lambda m: m.func, self.model, new_func)
                    print("Loaded normalization ranges from checkpoint.")

                start_step = metadata["step"]
                best_val_loss = metadata.get("best_val_loss", metadata.get("best_loss", float('inf')))
                best_train_loss = metadata.get("best_train_loss", float('inf'))
                print(f"Resumed from checkpoint at step {start_step} with best val loss {best_val_loss} and best train loss {best_train_loss}")
            except Exception as e:
                print(f"Failed to resume from checkpoint: {e}")

        master_key = jr.PRNGKey(self.seed)
        loader = self.dataloader(train_y0s, train_params, train_ys, batch_size=batch_size, key=master_key)

        best_model = self.model
        no_improve_counter = 0
        smooth_train_loss = None
        step_times = [] 
        early_stopping_info = None
        nan_divergence_counter = 0
        nan_divergence_threshold = training_config.get("nan_divergence_threshold", 50)  # consecutive NaN/diverged steps before terminating
        
        for step in range(start_step, max_steps):
            # Check for global timeout
            if timeout is not None and (time.time() - training_start_time) > timeout:
                print(f"Stopping early: Training time exceeded {timeout} seconds.")
                self.model = best_model
                early_stopping_info = {
                    "stopped_early": True,
                    "reason": "timeout",
                    "iteration": step,
                }
                break

            try:
                batch_y0, batch_params, batch_ys = next(loader)
                
                step_key, master_key = jr.split(master_key)

                step_start_time = time.time()
            
                self.model, opt_state, train_loss = self.train_step(
                    self.model,
                    opt_state,
                    ts,
                    batch_y0,
                    batch_params,
                    batch_ys,
                    optimizer,
                    step_key,
                    weight_decay
                )
                
                step_elapsed = time.time() - step_start_time
                step_times.append(step_elapsed)

                # First-step check
                if step == start_step:
                    if step_elapsed > first_step_timeout:
                        print(f"FAILED: First step took {step_elapsed:.1f}s, exceeds first_step_timeout={first_step_timeout}s.")
                        early_stopping_info = {
                            "stopped_early": True,
                            "reason": "first_step_timeout",
                            "iteration": step,
                        }
                        return losses, early_stopping_info, time.time() - training_start_time, best_val_loss, val_losses, best_train_loss
                    if float(train_loss) > divergence_loss_threshold:
                        print(f"FAILED: Loss after first step = {float(train_loss):.2e} > divergence_loss_threshold={divergence_loss_threshold:.0e}. ODE diverged.")
                        early_stopping_info = {
                            "stopped_early": True,
                            "reason": "diverged",
                            "iteration": step,
                        }
                        return losses, early_stopping_info, time.time() - training_start_time, best_val_loss, val_losses, best_train_loss

                # NaN / Divergence early termination
                if jnp.isnan(train_loss) or train_loss >= 1e8:
                    nan_divergence_counter += 1
                else:
                    nan_divergence_counter = 0
                
                if nan_divergence_counter >= nan_divergence_threshold:
                    print(f"FAILED: Model producing NaNs or Diverged for {nan_divergence_counter} consecutive steps. Terminating.")
                    early_stopping_info = {
                        "stopped_early": True,
                        "reason": "nan_diverged",
                        "iteration": step,
                    }
                    break

                losses.append(float(train_loss))

                # Only compute validation every print_freq steps. Doing this every step becomes very slow.
                if step % print_freq == 0 and n_val > 0:
                    val_model = eqx.nn.inference_mode(self.model)
                    val_pred = jax.vmap(lambda y0, p: val_model(ts, y0, p))(val_y0s, val_params)
                    val_pred_phys = val_pred * y_range + y_min
                    val_ys_phys = val_ys * y_range + y_min
                    mask = jnp.isfinite(val_pred_phys) & jnp.isfinite(val_ys_phys)
                    val_loss = float(jnp.mean(jnp.square(val_pred_phys[mask] - val_ys_phys[mask])))
                    val_losses.append(val_loss)
                elif not val_losses:
                    val_loss = -1.0
                else:
                    val_loss = val_losses[-1]

                if n_val > 0:
                   
                    if step % print_freq == 0 and step >= 500:
                        if val_loss < best_val_loss - min_delta:
                            best_val_loss = val_loss
                            best_model = self.model
                            no_improve_counter = 0
                        else:
                            no_improve_counter += print_freq

                        if no_improve_counter >= patience:
                            print(f"Early stopping (validation) at step {step}, best validation loss (after 500) = {best_val_loss:.6f}")
                            self.model = best_model
                            early_stopping_info = {
                                "stopped_early": True,
                                "reason": "val_loss_stagnation",
                                "iteration": step,
                            }
                            break
                else:
                    # Convergence check on smoothed training loss
                    if smooth_train_loss is None:
                        smooth_train_loss = float(train_loss)
                    else:
                        smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * float(train_loss)
                    
                    if smooth_train_loss < best_train_loss - min_delta:
                        best_train_loss = smooth_train_loss
                        best_model = self.model
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                        
                    if no_improve_counter >= patience:
                        print(f"Early stopping (training convergence) at step {step}, best smoothed loss = {best_train_loss:.6f}")
                        self.model = best_model
                        early_stopping_info = {
                            "stopped_early": True,
                            "reason": "loss_stagnation",
                            "iteration": step,
                        }
                        break

               
                if n_val > 0:
                     current_train_loss = float(train_loss)
                     if current_train_loss < best_train_loss:
                         best_train_loss = current_train_loss

                if step % print_freq == 0:
                    avg_step_time = np.mean(step_times[-min(100, len(step_times)):])
                    print(f"Step {step}/{max_steps}, train loss = {train_loss:.6f}, val loss = {val_loss:.6f}, step time = {step_elapsed:.2f}s (avg: {avg_step_time:.2f}s)")

                # Warning if step takes very long
                if step_elapsed > step_timeout_threshold:
                    print(f"WARNING: Step {step} took {step_elapsed:.1f}s (threshold: {step_timeout_threshold}s). Possible stuck ODE solve!")
                    
                if checkpoint_path and step % checkpoint_freq == 0:
                    self.save_checkpoint(checkpoint_path, step, opt_state, best_val_loss, best_train_loss)

            except KeyboardInterrupt:
                print("\nTraining interrupted by user (KeyboardInterrupt)!")
                early_stopping_info = {
                    "stopped_early": True,
                    "reason": "keyboard_interrupt",
                    "iteration": step if 'step' in locals() else 0,
                }
                break
        
        # Save final checkpoint if interrupted or finished
        if checkpoint_path and 'step' in locals():
             self.save_checkpoint(checkpoint_path, step, opt_state, best_val_loss, best_train_loss)

        total_training_time = time.time() - training_start_time     

        return losses, early_stopping_info, total_training_time, best_val_loss if n_val > 0 else best_train_loss, val_losses, best_train_loss
    
    
    def __call__(self, ts, y0, params):
      
        val_model = eqx.nn.inference_mode(self.model)

        if self.normalized:
            y_min = self.model.func.y_min
            y_range = self.model.func.y_range
            p_min = self.model.func.p_min
            p_range = self.model.func.p_range
            
            # Normalize
            y0_norm = (y0 - jnp.asarray(y_min)) / jnp.asarray(y_range)
            params_norm = (params - jnp.asarray(p_min)) / jnp.asarray(p_range)
            
            # Run model (returns normalized trajectory)
            ys_norm = val_model(ts, y0_norm, params_norm)
            
            # Denormalize output
            ys = ys_norm * jnp.asarray(y_range) + jnp.asarray(y_min)
            return jnp.maximum(ys, 0.0)
      
        else:
            return jnp.maximum(val_model(ts, y0, params), 0.0)

    def save(self, path: str):
       
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        eqx.tree_serialise_leaves(path / "model.eqx", self.model)
        
        metadata = {
            "filepath": str(self.filepath),
            "width_NODE": self.width_NODE,
            "depth_NODE": self.depth_NODE,
            "normalize": self.normalized,
            "dropout": self.dropout,
            "solver": self.solver,
            "dt0": self.dt0,
            "rtol": self.rtol,
            "atol": self.atol,
            "seed": self.seed,
            "y_min": list(self.model.func.y_min) if self.normalized else None,
            "y_range": list(self.model.func.y_range) if self.normalized else None,
            "p_min": list(self.model.func.p_min) if self.normalized else None,
            "p_range": list(self.model.func.p_range) if self.normalized else None
        }
        import json
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        print(f"Model saved to {path}")

    def save_checkpoint(self, path: str, step: int, opt_state, best_val_loss: float, best_train_loss: float):
        path = Path(f"{path}/step_{step}")
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        eqx.tree_serialise_leaves(path / "model.eqx", self.model)
        
        # Save optimizer state
        eqx.tree_serialise_leaves(path / "opt_state.eqx", opt_state)

        # Save metadata (including full architecture so load() can reconstruct correctly)
        metadata = {
            "step": step,
            "best_val_loss": float(best_val_loss),
            "best_train_loss": float(best_train_loss),
            "best_loss": float(best_val_loss),
            "filepath": str(self.filepath),
            "width_NODE": self.width_NODE,
            "depth_NODE": self.depth_NODE,
            "normalize": self.normalized,
            "dropout": self.dropout,
            "solver": self.solver,
            "dt0": self.dt0,
            "rtol": self.rtol,
            "atol": self.atol,
            "seed": self.seed,
            "y_min": list(self.model.func.y_min) if self.normalized else None,
            "y_range": list(self.model.func.y_range) if self.normalized else None,
            "p_min": list(self.model.func.p_min) if self.normalized else None,
            "p_range": list(self.model.func.p_range) if self.normalized else None
        }
        import json
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        print(f"Checkpoint saved to {path} at step {step}")

    def load_checkpoint(self, path: str, optimizer):
        path = Path(path)
        import json
        
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
            
        step = metadata["step"]
        best_val_loss = metadata.get("best_val_loss", metadata.get("best_loss", float('inf')))
        best_train_loss = metadata.get("best_train_loss", float('inf'))
        
        # Load model weights
        self.model = eqx.tree_deserialise_leaves(path / "model.eqx", self.model)
        
        if self.normalized and "y_min" in metadata and metadata["y_min"] is not None:
            old_func = self.model.func
            new_func = NODE_Func.__new__(NODE_Func)
            object.__setattr__(new_func, "nn_flux",    old_func.nn_flux)
            object.__setattr__(new_func, "y_min",      tuple(metadata["y_min"]))
            object.__setattr__(new_func, "y_range",    tuple(metadata["y_range"]))
            object.__setattr__(new_func, "p_min",      tuple(metadata["p_min"]))
            object.__setattr__(new_func, "p_range",    tuple(metadata["p_range"]))
            object.__setattr__(new_func, "normalized", True)
            object.__setattr__(new_func, "nn_scale",   old_func.nn_scale)
            self.model = eqx.tree_at(lambda m: m.func, self.model, new_func)

        return step, best_val_loss, best_train_loss

    @classmethod
    def load(cls, path: str):
       
        path = Path(path)
        import json
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
            
        # Backward compatibility for existing experiments
        # run_meta.json may be in the parent run_X directory when loading from a checkpoint step subdir
        run_meta = {}
        run_meta_path = path / "run_meta.json"
        if not run_meta_path.exists():
            # walk up to find run_meta.json (checkpoint subdirs have it in the parent run dir)
            for parent in path.parents:
                candidate = parent / "run_meta.json"
                if candidate.exists():
                    run_meta_path = candidate
                    break
        if run_meta_path.exists():
            with open(run_meta_path, "r") as f:
                run_meta = json.load(f)
        
        # Priority: metadata.json > run_meta.json > defaults
        width = metadata.get("width_NODE", run_meta.get("width", 32))
        depth = metadata.get("depth_NODE", run_meta.get("depth", 2))
        normalize = metadata.get("normalize", run_meta.get("normalize", True))
        dropout = metadata.get("dropout", run_meta.get("dropout", 0.1))
        solver = metadata.get("solver", run_meta.get("solver", "Kvaerno5"))
        dt0 = metadata.get("dt0", run_meta.get("dt0", 1e-4))
        rtol = metadata.get("rtol", run_meta.get("rtol", 1e-4))
        atol = metadata.get("atol", run_meta.get("atol", 1e-6))
        seed = metadata.get("seed", 0)
        
        model_config = {
            "width_NODE": width,
            "depth_NODE": depth,
            "normalize": normalize,
            "dropout": dropout,
            "solver": solver,
            "dt0": dt0,
            "rtol": rtol,
            "atol": atol,
            "seed": seed
        }
        
        instance = cls(metadata["filepath"], 
                       mask=None,
                       model_config=model_config)
        
        instance.model = eqx.tree_deserialise_leaves(path / "model.eqx", instance.model)
        
        if instance.normalized:
            y_min = metadata.get("y_min", run_meta.get("y_min", [0.0]*instance.n_metabolites))
            y_range = metadata.get("y_range", run_meta.get("y_range", [1.0]*instance.n_metabolites))
            p_min = metadata.get("p_min", run_meta.get("p_min", [0.0]*instance.n_params))
            p_range = metadata.get("p_range", run_meta.get("p_range", [1.0]*instance.n_params))
            if y_min is not None:
                old_func = instance.model.func
                new_func = NODE_Func.__new__(NODE_Func)
                object.__setattr__(new_func, "nn_flux",    old_func.nn_flux)
                object.__setattr__(new_func, "y_min",      tuple(y_min))
                object.__setattr__(new_func, "y_range",    tuple(y_range))
                object.__setattr__(new_func, "p_min",      tuple(p_min))
                object.__setattr__(new_func, "p_range",    tuple(p_range))
                object.__setattr__(new_func, "normalized", True)
                object.__setattr__(new_func, "nn_scale",   old_func.nn_scale)
                instance.model = eqx.tree_at(lambda m: m.func, instance.model, new_func)

        return instance


class Neural_ODE(eqx.Module):
    layers: tuple
    dropout: eqx.nn.Dropout

    def __init__(self, state_dim, param_dim, width, depth, key, dropout):
        keys = jr.split(key, depth + 1)
        layers = []
        
        layers.append(eqx.nn.Linear(state_dim + param_dim, width, key=keys[0]))
        layers.append(jnn.tanh)
        
        for i in range(depth - 1):
            layers.append(eqx.nn.Linear(width, width, key=keys[i+1]))
            layers.append(jnn.tanh)
        
        # Output: dydt directly (size state_dim)
        layers.append(eqx.nn.Linear(width, state_dim, key=keys[-1]))
        
        # Near zero weight initialization
        def scale_linear(layer):
            if isinstance(layer, eqx.nn.Linear):
                if hasattr(layer, "weight"):
                    layer = eqx.tree_at(lambda l: l.weight, layer, layer.weight * 1e-2)
                    if hasattr(layer, "bias") and layer.bias is not None:
                        layer = eqx.tree_at(lambda l: l.bias, layer, jnp.zeros_like(layer.bias))
            return layer

        self.layers = tuple(scale_linear(l) for l in layers)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, y, params, key=None):
        x = jnp.concatenate([y, params])
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, eqx.nn.Linear):
                x = layer(x)
                if i < len(self.layers) - 2:
                    if not self.dropout.inference:
                        key, subkey = jr.split(key)
                        x = self.dropout(x, key=subkey)
            else:
                x = layer(x)
        return x

class NODE_Func(eqx.Module):
    nn_flux: Neural_ODE
    y_min: jnp.ndarray = eqx.field(static=True)
    y_range: jnp.ndarray = eqx.field(static=True)
    p_min: jnp.ndarray = eqx.field(static=True)
    p_range: jnp.ndarray = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    nn_scale: jnp.ndarray

    def __init__(self, state_dim, param_dim, width, depth, key, dropout, normalized):
        self.nn_flux = Neural_ODE(state_dim, param_dim, width, depth, key, dropout)
        self.normalized = normalized

        self.y_min = tuple(np.zeros(state_dim).tolist())
        self.y_range = tuple(np.ones(state_dim).tolist())
        self.p_min = tuple(np.zeros(param_dim).tolist())
        self.p_range = tuple(np.ones(param_dim).tolist())

        self.nn_scale = jnp.array(1e-1)

    def __call__(self, t, y, args):
        params, key = args
        
        if self.normalized:

            y_safe = jnn.relu(y)
          
            dydt_norm = self.nn_scale * self.nn_flux(y_safe, params, key=key)
            dydt_norm = jnp.clip(dydt_norm, -50.0, 50.0)
            negative_dydt_mask = dydt_norm < 0

            epsilon = 1e-4
            damping_factor = y_safe / (y_safe + epsilon)
            
            dydt_norm = jnp.where(negative_dydt_mask, dydt_norm * damping_factor, dydt_norm)
            return dydt_norm
        else:
            y_safe = jnn.relu(y)
            # Predict dydt directly
            dydt = 1e-3 * self.nn_flux(y_safe, params, key=key)

            negative_dydt_mask = dydt < 0

            epsilon = 1e-4
            damping_factor = y_safe / (y_safe + epsilon)
            
            dydt = jnp.where(negative_dydt_mask, dydt * damping_factor, dydt)
            return dydt

class NODE_Integrator(eqx.Module):
    func: NODE_Func
    solver_name: str
    dt0: float
    rtol: float
    atol: float

    def __init__(self, state_dim, param_dim, width, depth, key, dropout, normalized, solver, dt0, rtol, atol):
        self.func = NODE_Func(state_dim, param_dim, width, depth, key, dropout, normalized)
        self.solver_name = solver
        self.dt0 = dt0
        self.rtol = rtol
        self.atol = atol

    def __call__(self, ts, y0, params, key=None):

        def event_fn(t, y, args, **kwargs):
            return jnp.any(y < -10) | jnp.any(y > 1e7)
        
        term = diffrax.ODETerm(eqx.Partial(self.func))
        
        # Decide on stepsize controller based on solver
        if self.solver_name in ["Euler"]:
            # Fixed-step solvers don't provide error estimates for PIDController
            stepsize_controller = diffrax.ConstantStepSize()
        else:
            # Adaptive solvers (Kvaerno, Tsit, Dopri) use PIDController
            stepsize_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol)

        sol = diffrax.diffeqsolve(
            term,
            getattr(diffrax, self.solver_name)(),
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt0,
            y0=y0,
            args= (params, key),
            saveat=diffrax.SaveAt(ts=ts), 
            stepsize_controller=stepsize_controller,
            max_steps=4000,
            event=diffrax.Event(event_fn),
            throw=False,
        )
        return sol.ys