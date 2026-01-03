"""
Bayesian trainable class for RNNAssembly experiments.
This file is specifically created for Bayesian search experiments and uses:
1. AdamW optimizer with weight decay
2. Enhanced RNNAssembly with LayerNorm and attention pooling
3. Improved learning rate scheduling

Added for Bayesian search experiments - does not modify existing codebase.
"""

from typing import Optional, Dict
from ray import tune
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, 
    OneCycleLR,
    ReduceLROnPlateau
)
import math

from common_utils import (
    load_mnist, load_har2_dataloaders, get_har2_input_output_sizes, 
    load_forda_dataloaders, get_forda_input_output_sizes,
    load_fordb_dataloaders, get_fordb_input_output_sizes, 
    load_adiac_dataloaders, get_adiac_input_output_sizes,
    load_IEEEPPG_dataloaders, get_IEEEPPG_input_output_sizes,
    load_NewsTitleSentiment_dataloaders, get_NewsTitleSentiment_input_output_sizes,
    load_JapaneseVowels_dataloaders, get_JapaneseVowels_input_output_sizes,
    load_PEMS_SF_dataloaders, get_PEMS_SF_input_output_sizes,
)

from enhanced_rnn_assembly import EnhancedRNNAssembly
from trainable import get_block_config

# Added for Bayesian search: fixed version of get_init_fn to handle uniform function correctly
def get_init_fn_fixed(fn_str: str, *args):
    """Fixed version of get_init_fn for Bayesian search."""
    from torchdyno.models.initializers import sparse, diagonal, orthogonal, zeros, uniform
    import torch

    if fn_str == "sparse":
        return lambda x, dtype: sparse(x, *args, dtype=dtype)
    if fn_str == "orthogonal":
        return lambda x, dtype: orthogonal(x, dtype=dtype)
    if fn_str == "diagonal":
        return lambda x, dtype: diagonal(x, dtype=dtype)
    if fn_str == "lt":
        return lambda x, dtype: diagonal(x, dtype=dtype) + torch.tril(uniform(x, dtype=dtype))
    if fn_str == "uniform":
        return lambda x, dtype: uniform(x, dtype=dtype)  # Fixed: removed None, None arguments
    if fn_str == "zeros":
        return lambda x, dtype: zeros(x, dtype=dtype)
    raise ValueError(f"Unknown initialization function {fn_str}")

regression_datasets = ["ieeeppg", "newstitlesentiment"]


class BayesianRNNAssemblyTrainable(tune.Trainable):
    """Enhanced trainable class for Bayesian search experiments."""

    def setup(self, config: Dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.check_stability = config.get("check_stability", False)
        self.config = config
        self.build_trial(config)
        
        eff_params = 0
        if hasattr(self.model, "count_effective_trainable_parameters"):
            eff_params = self.model.count_effective_trainable_parameters()
            print(f"Trainable parameters (effective, with Bayesian enhancements): {eff_params}")
        else:
            eff_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable parameters (total, with Bayesian enhancements): {eff_params}")

    def step(self):
        run_loss_t = 0.0
        run_acc_t = 0.0
        n_batches = 0
        self.model.train()
        from tqdm import tqdm

        for x, y in tqdm(self.train_loader, desc="Batches processed", unit="batch"):
            self.opt.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            
            all_pred, _ = self.model(x)
            
            if self.model.use_attention_pooling:
                # For attention pooling, all_pred is the direct output from attention mechanism
                y_pred = all_pred
            else:
                # For standard pooling, use the last time step
                y_pred = all_pred[-1]

            loss = self.loss(y_pred, y)
            loss.backward()
            self.opt.step()

            run_loss_t += loss.item()
            if self.config.get("dataset", "mnist") not in regression_datasets:
                run_acc_t += (
                    (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item()
                ) / len(y)
            n_batches += 1
            
        run_loss_t /= n_batches
        if self.config.get("dataset", "mnist") not in regression_datasets:
            run_acc_t /= n_batches
            
        if hasattr(self, 'lr_scheduler'):
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                pass
            else:
                self.lr_scheduler.step()

        stability_metrics = {}
        if self.check_stability:
            with torch.no_grad():
                coupling_matrix = self.model._couplings.couplings
                eigvals = torch.linalg.eigvals(coupling_matrix)
                spectral_radius = eigvals.abs().max().item()
                spectral_norm = torch.linalg.norm(coupling_matrix, ord=2).item()
                stability_metrics["spectral_radius"] = spectral_radius
                stability_metrics["spectral_norm"] = spectral_norm

        self.model.eval()
        run_loss_e = 0.0
        run_acc_e = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                all_pred, _ = self.model(x)
                
                if self.model.use_attention_pooling:
                    # For attention pooling, all_pred is the direct output from attention mechanism
                    y_pred = all_pred
                else:
                    # For standard pooling, use the last time step
                    y_pred = all_pred[-1]

                loss = self.loss(y_pred, y)
                run_loss_e += loss.item()
                if self.config.get("dataset", "mnist") not in regression_datasets:
                    run_acc_e += (
                        (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item()
                    ) / len(y)
                n_batches += 1
                
        run_loss_e /= n_batches
        if self.config.get("dataset", "mnist") not in regression_datasets:
            run_acc_e /= n_batches
            
        if hasattr(self, 'lr_scheduler') and isinstance(self.lr_scheduler, ReduceLROnPlateau):
            self.lr_scheduler.step(run_loss_e)

        if self.config.get("dataset", "mnist") in regression_datasets:
            res = {
                "train_loss": run_loss_t,
                "train_RMSE": math.sqrt(run_loss_t),
                "test_loss": run_loss_e,
                "test_RMSE": math.sqrt(run_loss_e),
                "n_trainable_params": self.model.count_effective_trainable_parameters(),
                "lr": self.config["lr"],
                "weight_decay": self.config.get("weight_decay", 0.0),
            }
        else:
            res = {
                "train_loss": run_loss_t,
                "train_acc": run_acc_t,
                "test_loss": run_loss_e,
                "test_acc": run_acc_e,
                "n_trainable_params": self.model.count_effective_trainable_parameters(),
                "lr": self.config["lr"],
                "weight_decay": self.config.get("weight_decay", 0.0),
            }
            
        if self.check_stability:
            res.update(stability_metrics)
            
        return res

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict]:
        self.model.to("cpu")
        torch.save(self.model, f"{checkpoint_dir}/model.pth")
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint: Dict):
        self.model = torch.load(checkpoint["model.pth"])
        self.model.to(self.device)

    def build_trial(self, config):
        dataset = config.get("dataset", "mnist")
        
        if dataset == "har2":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_har2_dataloaders(
                root=config["root"],
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
            input_size, out_size = get_har2_input_output_sizes()
        elif dataset == "forda":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_forda_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
            input_size, out_size = get_forda_input_output_sizes()
        elif dataset == "fordb":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_fordb_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
            input_size, out_size = get_fordb_input_output_sizes()
        elif dataset == "adiac":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_adiac_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
            input_size, out_size = get_adiac_input_output_sizes()
        elif dataset == "japanesevowels":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_JapaneseVowels_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
            input_size, out_size = get_JapaneseVowels_input_output_sizes()
        elif dataset == "ieeeppg":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_IEEEPPG_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
            input_size, out_size = get_IEEEPPG_input_output_sizes()
        elif dataset == "newstitlesentiment":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_NewsTitleSentiment_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
            input_size, out_size = get_NewsTitleSentiment_input_output_sizes()
        elif dataset in ["smnist", "psmnist"]:
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader = load_mnist(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size,
                permute_seed=config["permute_seed"],
                root=config["root"]
            )
            input_size = 1
            out_size = 10
        elif dataset == "pems":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_PEMS_SF_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size,
            )
            input_size, out_size = get_PEMS_SF_input_output_sizes()
        else:
            raise ValueError(f"Dataset {dataset} not supported for BayesianRNNAssemblyTrainable.")

        block_config = get_block_config(config["block_config"])
        dtype = torch.float32

        model_params = {
            "input_size": input_size,
            "out_size": out_size,
            "block_sizes": config["block_sizes"],
            "block_init_fn": get_init_fn_fixed(*block_config[0]),
            "coupling_block_init_fn": get_init_fn_fixed(*config["coupling_block_init_fn"]),
            "coupling_topology": config["coupling_topology"],
            "eul_step": config["eul_step"],
            "activation": config["activation"],
            "constrained_blocks": block_config[1],
            "dtype": dtype,
            "gated_eul": config["gating"],
            "min_gate": config["gate_bounds"][0],
            "max_gate": config["gate_bounds"][1],
            "coupling_rescaling": config["coupling_rescaling"],
            "spectral_threshold": None,
            "structure": config.get("block_structure", "identity"),
            "use_layer_norm": config.get("use_layer_norm", True),
            "use_attention_pooling": config.get("use_attention_pooling", True),
        }

        print(f"Using EnhancedRNNAssembly with LayerNorm: {model_params['use_layer_norm']}, "
              f"Attention pooling: {model_params['use_attention_pooling']}")
        
        self.model = EnhancedRNNAssembly.from_initializers(**model_params)
        self.model.to(device=self.device)

        if dataset in regression_datasets:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        weight_decay = config.get("weight_decay", 0.0)
        self.opt = AdamW(
            self.model.parameters(), 
            lr=config["lr"],
            weight_decay=weight_decay
        )
        print(f"Using AdamW optimizer with lr={config['lr']}, weight_decay={weight_decay}")

        scheduler_type = config.get("lr_scheduler", "cosine")
        max_epochs = config.get("max_epochs", 200)
        
        if scheduler_type == "cosine":
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.opt,
                T_0=50,
                T_mult=2,
                eta_min=1e-6
            )
        elif scheduler_type == "onecycle":
            steps_per_epoch = len(self.train_loader)
            self.lr_scheduler = OneCycleLR(
                self.opt,
                max_lr=config["lr"],
                epochs=max_epochs,
                steps_per_epoch=steps_per_epoch
            )
        elif scheduler_type == "plateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.opt,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        else:
            self.lr_scheduler = None
            
        print(f"Using learning rate scheduler: {scheduler_type}")