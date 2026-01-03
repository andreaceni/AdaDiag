from typing import Optional
from typing import Dict
from ray import tune
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchdyno.models.initializers import uniform
import math

from common_utils import (load_mnist, load_har2_dataloaders, get_har2_input_output_sizes, load_forda_dataloaders, 
                          get_forda_input_output_sizes, 
                          load_fordb_dataloaders, get_fordb_input_output_sizes, load_adiac_dataloaders, get_adiac_input_output_sizes)
from common_utils import load_IEEEPPG_dataloaders, get_IEEEPPG_input_output_sizes, load_NewsHeadlineSentiment_dataloaders, get_NewsHeadlineSentiment_input_output_sizes, load_AppliancesEnergy_dataloaders, get_AppliancesEnergy_input_output_sizes, load_NewsTitleSentiment_dataloaders, get_NewsTitleSentiment_input_output_sizes
from common_utils import load_JapaneseVowels_dataloaders, get_JapaneseVowels_input_output_sizes
from common_utils import (
    load_PEMS_SF_dataloaders, get_PEMS_SF_input_output_sizes,
)

regression_datasets = [ "ieeeppg","newstitlesentiment"]

from torchdyno.data.datasets.mg17 import get_mackey_glass
from torchdyno.data.datasets.motionHAR import get_motion_data

from torchdyno.models.rnn_assembly import RNNAssembly
from rnn_assembly_modular import RNNAssemblyModular


class RNNAssemblyTrainable(tune.Trainable):

    def setup(self, config: Dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.check_stability = config.get("check_stability", False)
        self.config = config  # Store config for use in step() method
        self.build_trial(config)
        eff_params = 0
        if hasattr(self.model, "count_effective_trainable_parameters"):
            eff_params = self.model.count_effective_trainable_parameters()
            print("Trainable parameters (effective):", eff_params)
        else:
            eff_params = self.model.count_trainable_parameters()
            print("Trainable parameters: ", eff_params)


    def step(self):
        run_loss_t = 0.0
        run_acc_t = 0.0
        n_batches = 0
        self.model.train()
        from tqdm import tqdm

        #####################################
        # added for storing frequencies
        import os
        store_frequencies = self.config.get("store_frequencies", False)
        if store_frequencies and not getattr(self.model, "gated_eul", False):
            raise RuntimeError("store_frequencies=True requires gated_eul=True in the model.")

        # Robust epoch index persistence in output directory
        out_dir = '/storagenfs/a052721/AdaDiag/final_search/fordb/rnnassembly/frequencies/'
        #out_dir = self.config.get("results_dir", "./")
        os.makedirs(out_dir, exist_ok=True)
        epoch_idx_file = os.path.join(out_dir, "gated_steps_epoch_idx.txt")
        if os.path.exists(epoch_idx_file):
            with open(epoch_idx_file, "r") as f:
                epoch_idx = int(f.read().strip())
        else:
            epoch_idx = 0
        gated_steps_epoch = []
        trace_firstbatch = True
        #####################################

        for x, y in tqdm(self.train_loader, desc="Batches processed", unit="batch"):
            self.opt.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            #####################################
            #print('Store frequencies:', store_frequencies)
            # if-branch added for storing frequencies
            if trace_firstbatch: # trace only the first batch of each epoch to avoid excessive memory usage
                if store_frequencies and hasattr(self.model, "compute_states"):
                    states = []
                    gated_steps = []
                    device = self.model._input_mat.device
                    state = torch.zeros(self.model.hidden_size, dtype=self.model._dtype, device=device)
                    timesteps = x.shape[0]
                    for t in range(timesteps):
                        xt = x[t].to(device)
                        if self.model._blocks._constrained == "tanh_input_state":
                            blocks_out = self.model._blocks(self.model.activ_fn(state), current_input=xt, current_state=state)
                        else:
                            blocks_out = self.model._blocks(self.model.activ_fn(state))
                        if isinstance(blocks_out, torch.Tensor):
                            blocks_out = blocks_out.to(state.device)
                        couplings_out = self.model._couplings(state).to(state.device)
                        input_out = torch.nn.functional.linear(xt, self.model._input_mat).to(state.device)
                        gated_step = self.model.compute_gate(xt, state)
                        #print('Gated step shape at each time step:', gated_step.detach().cpu().shape)
                        # take the mean over the batch dimension
                        gated_step_mean = gated_step.mean(dim=0)
                        gated_steps.append(gated_step_mean.detach().cpu())
                        #print('Gated step shape at each time step:', gated_step_mean.detach().cpu().shape)
                        state = state + gated_step * (
                            -state + blocks_out + couplings_out + input_out
                        )
                        states.append(state)
                    gated_steps_epoch.append(torch.stack(gated_steps, dim=0))
                    all_pred = [states]
                    y_pred = states[-1]
                else: 
                    all_pred, _ = self.model(x)
                    y_pred = all_pred[-1]
                
                trace_firstbatch = False # this allows to store frequencies only for the first batch of each epoch
            else:
                #####################################
                all_pred, _ = self.model(x)
                y_pred = all_pred[-1]

            loss = self.loss(y_pred, y)
            loss.backward()
            self.opt.step()

            run_loss_t += loss.item()
            # compute accuracy only for classification tasks
            if self.config.get("dataset", "mnist") not in regression_datasets:
                run_acc_t += (
                    (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item()
                ) / len(y)
            n_batches += 1
        run_loss_t /= n_batches
        # Compute accuracy only for classification tasks
        if self.config.get("dataset", "mnist") not in regression_datasets:
            run_acc_t /= n_batches
        self.lr.step()

        #####################################
        # added for storing frequencies
        print('Length of gated_steps_epoch:', len(gated_steps_epoch))
        # Save after each epoch if requested
        if store_frequencies and len(gated_steps_epoch) > 0:
            torch.save(gated_steps_epoch, os.path.join(out_dir, f"gated_steps_epoch_{epoch_idx}.pt"))
            print('Stored frequencies for epoch', epoch_idx, ' at ', os.path.join(out_dir, f"gated_steps_epoch_{epoch_idx}.pt"))
        # Increment epoch_idx for next call if running in a loop
        if store_frequencies:
            # Persist the incremented epoch index for next epoch
            with open(epoch_idx_file, "w") as f:
                f.write(str(epoch_idx + 1))
        #####################################


        # Stability check
        stability_metrics = {}
        if self.check_stability:
            with torch.no_grad():
                coupling_matrix = self.model._couplings.couplings
                # Spectral radius (largest absolute eigenvalue)
                eigvals = torch.linalg.eigvals(coupling_matrix)
                spectral_radius = eigvals.abs().max().item()
                # Spectral norm (largest singular value)
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
                
                y_pred = all_pred[-1]

                loss = self.loss(y_pred, y)
                run_loss_e += loss.item()
                # compute accuracy only for classification tasks
                if self.config.get("dataset", "mnist") not in regression_datasets:
                    run_acc_e += (
                        (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item()
                    ) / len(y)
                n_batches += 1
        run_loss_e /= n_batches
        # Compute accuracy only for classification tasks
        if self.config.get("dataset", "mnist") not in regression_datasets:
            run_acc_e /= n_batches
        
        # For regression tasks, log MSE and RMSE instead of accuracy
        if self.config.get("dataset", "mnist") in regression_datasets:
            res = {
                "train_loss": run_loss_t, # this is MSE
                "train_RMSE": math.sqrt(run_loss_t),
                "test_loss": run_loss_e, # this is MSE
                "test_RMSE": math.sqrt(run_loss_e),
                "n_trainable_params": self.model.count_effective_trainable_parameters(),
                "lr": self.config["lr"],
            }
        else:
            res = {
                "train_loss": run_loss_t,
                "train_acc": run_acc_t,
                "test_loss": run_loss_e,
                "test_acc": run_acc_e,
                "n_trainable_params": self.model.count_effective_trainable_parameters(),
                "lr": self.config["lr"],
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
        # Build model with configuration

        # Dataset selection logic
        if config.get("dataset", "mnist") == "har":
            # har returns (train, val, test) tuples
            (train_data, train_targets), (val_data, val_targets), _ = get_motion_data(train_batch_size=120,test_batch_size=500)
            train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
            val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=120, shuffle=True)
            self.eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=500, shuffle=False)
        elif config.get("dataset", "mnist") == "har2":
            # HAR-2 binary classification dataset
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_har2_dataloaders(
                root=config["root"],
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
        elif config.get("dataset", "mnist") == "forda":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_forda_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
        elif config.get("dataset", "mnist") == "fordb":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_fordb_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
        elif config.get("dataset", "mnist") == "adiac":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_adiac_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
        elif config.get("dataset", "mnist") == "japanesevowels":
            train_batch_size = config.get("train_batch_size", 64)
            eval_batch_size = config.get("eval_batch_size", 64) 
            self.train_loader, self.eval_loader, _ = load_JapaneseVowels_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
        elif config.get("dataset", "mnist") == "ieeeppg":
            train_batch_size = config.get("train_batch_size", 64)   
            eval_batch_size = config.get("eval_batch_size", 64)     
            self.train_loader, self.eval_loader, _ = load_IEEEPPG_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
        elif config.get("dataset", "mnist") == "newstitlesentiment":
            train_batch_size = config.get("train_batch_size", 64)   
            eval_batch_size = config.get("eval_batch_size", 64)     
            self.train_loader, self.eval_loader, _ = load_NewsTitleSentiment_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size
            )
        elif config.get("dataset", "mnist") == "smnist":
            train_batch_size = config.get("train_batch_size", 64)   
            eval_batch_size = config.get("eval_batch_size", 64)     
            self.train_loader, self.eval_loader = load_mnist(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size,
                permute_seed=config["permute_seed"], 
                root=config["root"]
            )
        elif config.get("dataset", "mnist") == "psmnist":
            train_batch_size = config.get("train_batch_size", 64)   
            eval_batch_size = config.get("eval_batch_size", 64)     
            self.train_loader, self.eval_loader = load_mnist(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size,
                permute_seed=config["permute_seed"], 
                root=config["root"]
            )
        elif config.get("dataset", "mnist") == "pems":
            train_batch_size = config.get("train_batch_size", 64)   
            eval_batch_size = config.get("eval_batch_size", 64)
            self.train_loader, self.eval_loader, _ = load_PEMS_SF_dataloaders(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size,
            )
        else:
            dtst = config.get("dataset", "mnist")
            raise ValueError(f"Dataset {dtst} not supported for RNNAssemblyTrainable.")
        
        block_config = get_block_config(config["block_config"])
        # Force dtype to float32
        if config["block_config"] in [1, 2, 3, 13]:
            dtype = torch.float32

        # Set the spectral threshold for the coupling matrix
        spectral_threshold = None # corresponds to "unconstrained".


        # Set input/output sizes based on dataset
        if config.get("dataset", "mnist") == "har2":
            input_size, out_size = get_har2_input_output_sizes()
        elif config.get("dataset", "mnist") == "forda":
            input_size, out_size = get_forda_input_output_sizes()
        elif config.get("dataset", "mnist") == "ieeeppg":
            input_size, out_size = get_IEEEPPG_input_output_sizes()
        elif config.get("dataset", "mnist") == "newstitlesentiment":
            input_size, out_size = get_NewsTitleSentiment_input_output_sizes()
        elif config.get("dataset", "mnist") == "fordb":
            input_size, out_size = get_fordb_input_output_sizes()
        elif config.get("dataset", "mnist") == "adiac":
            input_size, out_size = get_adiac_input_output_sizes()
        elif config.get("dataset", "mnist") == "japanesevowels":
            input_size, out_size = get_JapaneseVowels_input_output_sizes()
        elif config.get("dataset", "mnist") in ["smnist", "psmnist"]:   
            input_size = 1
            out_size = 10
        elif config.get("dataset", "mnist") == "pems":
            input_size, out_size = get_PEMS_SF_input_output_sizes()
        else: # raise error
            dtst = config.get("dataset", "mnist")
            raise ValueError(f"Dataset {dtst} not supported for RNNAssemblyTrainable.")

        # Select model based on configuration mode
        model_mode = config.get("model_mode", "standard")
        
        # Common parameters for all model types
        model_params = {
            "input_size": input_size,
            "out_size": out_size,
            "block_sizes": config["block_sizes"],
            "block_init_fn": get_init_fn(*block_config[0]),
            "coupling_block_init_fn": get_init_fn(*config["coupling_block_init_fn"]),
            "coupling_topology": config["coupling_topology"],
            "eul_step": config["eul_step"],
            #"gamma": config["gamma"], # <--------------------------- TO DELETE?
            "activation": config["activation"],
            "constrained_blocks": block_config[1],
            "dtype": dtype,
            "gated_eul": config["gating"],  # Use gated Euler step if True, otherwise use fixed Euler step
            "min_gate": config["gate_bounds"][0], # minimum gate step
            "max_gate": config["gate_bounds"][1],  # maximum gate step
            "coupling_rescaling": config["coupling_rescaling"],  # "local", "global", or "unconstrained" # <--------------------------- TO DELETE?
            "spectral_threshold": spectral_threshold,  # Pass spectral constraint # <--------------------------- TO DELETE?
            "structure": config.get("block_structure", "identity"),  # Pass block structure to RNNAssembly # <--------------------------- TO DELETE?
        }
        
        if model_mode == "modular":
            # Use the new modular model with one channel per module
            print(f"Using RNNAssemblyModular (one channel per module)")
            self.model = RNNAssemblyModular.from_initializers(**model_params)
        else:
            # Use standard model (all modules see all channels)
            print(f"Using standard RNNAssembly (all modules see all channels)")
            self.model = RNNAssembly.from_initializers(**model_params)
        self.model.to(device=self.device)

        # Use MSELoss for regression (floodmodeling), CrossEntropyLoss for classification tasks (MNIST, HAR-2)
        if config.get("dataset", "mnist") in regression_datasets:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.opt = Adam(self.model.parameters(), lr=config["lr"]) 

        extended_search = True
        if extended_search:
            # Cosine Annealing LR scheduler
            # (this scheduler is used for the extended model selection of AdaDiag)
            self.lr = CosineAnnealingWarmRestarts(
                self.opt,
                T_0=50,          # Number of epochs for the first cycle
                T_mult=2,        # Cycle length multiplier for subsequent cycles
                eta_min=1e-5
            ) 
        else:
            # MultiStepLR scheduler with milestones and decay scalar from config 
            # (this scheduled has been used for model selection involving LSTM, VanillaRNN, SCN,and AdaDiag)
            self.lr = MultiStepLR(
                self.opt,
                milestones=config["decay_epochs"],
                gamma=config["decay_scalar"],
            )






def get_init_fn(fn_str: str, *args):
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
        return lambda x, dtype: uniform(x, None, None, dtype=dtype)
    if fn_str == "zeros":
        return lambda x, dtype: zeros(x, dtype=dtype)
    raise ValueError(f"Unknown initialization function {fn_str}")


def get_block_config(idx: int):
    # if idx == 0:
    #     return (("orthogonal",), "orthogonal")
    if idx == 1:
        return (("sparse", 0.03), "fixed")
    if idx == 2:
        return (("sparse", 0.1), "fixed")
    if idx == 3:
        # implements diagonal as tanh(b)
        return (("diagonal",), "tanh")
    if idx == 13:
        # implements diagonal as tanh(V x_{t+1} + W_i h_{i,t} + b)
        return (("diagonal",), "tanh_input_state") # input+state-dependent diagonal without 1-rank, W_i diagonal matrices
    else:
        raise ValueError(f"Unknown block configuration index: {idx}")






# === LSTM Trainable class for comparison experiments === #


class LSTMTrainable(tune.Trainable):
    def setup(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.build_trial(config)

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict]:
        self.model.to("cpu")
        torch.save(self.model, f"{checkpoint_dir}/model.pth")
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint: Dict):
        self.model = torch.load(checkpoint["model.pth"])
        self.model.to(self.device)


    def build_trial(self, config):
        # Dataset loading (reuse logic from RNNAssemblyTrainable)
        dataset = config.get("dataset", "mnist")
        train_batch_size = config.get("train_batch_size", 64)
        eval_batch_size = config.get("eval_batch_size", 64)
        root = config.get("root", None)
        if dataset == "har2":
            self.train_loader, self.eval_loader, _ = load_har2_dataloaders(
                root=root, train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_har2_input_output_sizes()
        elif dataset == "forda":
            self.train_loader, self.eval_loader, _ = load_forda_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_forda_input_output_sizes()
        elif dataset == "adiac":
            self.train_loader, self.eval_loader, _ = load_adiac_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_adiac_input_output_sizes()
        elif dataset == "fordb":
            self.train_loader, self.eval_loader, _ = load_fordb_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_fordb_input_output_sizes()
        elif dataset == "japanesevowels":
            self.train_loader, self.eval_loader, _ = load_JapaneseVowels_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_JapaneseVowels_input_output_sizes()
        elif dataset == "newstitlesentiment":
            self.train_loader, self.eval_loader, _ = load_NewsTitleSentiment_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_NewsTitleSentiment_input_output_sizes()
        elif dataset == "ieeeppg":
            self.train_loader, self.eval_loader, _ = load_IEEEPPG_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_IEEEPPG_input_output_sizes()
        elif dataset in ["smnist", "psmnist"]:   
            self.train_loader, self.eval_loader = load_mnist(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size,
                permute_seed=config["permute_seed"], 
                root=config["root"]
            )
            input_size = 1
            out_size = 10
        elif dataset == "pems":
            self.train_loader, self.eval_loader, _ = load_PEMS_SF_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size, 
            )
            input_size, out_size = get_PEMS_SF_input_output_sizes()
        else:
            raise ValueError(f"Dataset {dataset} not supported for LSTMTrainable.")

        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("n_layers", 1),
            bidirectional=config.get("bidirectional", False),
            batch_first=False
        ).to(self.device)
        self.classifier = nn.Linear(
            config.get("hidden_size", 128) * (2 if config.get("bidirectional", False) else 1),
            out_size
        ).to(self.device)
        # Use CrossEntropyLoss for classification, otherwise MSE
        if dataset in regression_datasets:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.opt = Adam(
            list(self.model.parameters()) + list(self.classifier.parameters()),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("l2_reg", 0.0)
        )
        self.lr = MultiStepLR(self.opt, milestones=[80, 160], gamma=0.1)

    def step(self):
        self.model.train()
        self.classifier.train()
        run_loss_t = 0.0
        run_acc_t = 0.0
        n_batches = 0
        self.model.train()
        #""" if you want the progress bar, uncomment the following lines
        from tqdm import tqdm
        for x, y in tqdm(self.train_loader, desc="Batches processed", unit="batch"):
        #"""
        #for x, y in self.train_loader:
            self.opt.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            all_pred, _ = self.model(x)
            
            y_pred = all_pred[-1] 
            y_pred = self.classifier(y_pred)

            loss = self.loss(y_pred, y)
            loss.backward()
            self.opt.step()

            run_loss_t += loss.item()
            # compute accuracy only for classification tasks
            if self.config.get("dataset", "mnist") not in regression_datasets:
                run_acc_t += (
                    (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item()
                ) / len(y)
            n_batches += 1
        run_loss_t /= n_batches
        # Compute accuracy only for classification tasks
        if self.config.get("dataset", "mnist") not in regression_datasets:
            run_acc_t /= n_batches
        self.lr.step()

        self.model.eval()
        self.classifier.eval()
        run_loss_e = 0.0
        run_acc_e = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                all_pred, _ = self.model(x)
                
                y_pred = all_pred[-1]
                y_pred = self.classifier(y_pred)

                loss = self.loss(y_pred, y)
                run_loss_e += loss.item()
                # compute accuracy only for classification tasks
                if self.config.get("dataset", "mnist") not in regression_datasets:
                    run_acc_e += (
                        (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item()
                    ) / len(y)
                n_batches += 1
        run_loss_e /= n_batches
        # Compute accuracy only for classification tasks
        if self.config.get("dataset", "mnist") not in regression_datasets:
            run_acc_e /= n_batches

        # For regression tasks, log MSE and RMSE instead of accuracy
        if self.config.get("dataset", "mnist") in regression_datasets:
            res = {
                "train_loss": run_loss_t, # this is MSE
                "train_RMSE": math.sqrt(run_loss_t),
                "test_loss": run_loss_e, # this is MSE
                "test_RMSE": math.sqrt(run_loss_e),
                "lr": self.config["lr"],
                "n_trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad) + sum(p.numel() for p in self.classifier.parameters() if p.requires_grad),
            }
        else:
            res = {
                "train_loss": run_loss_t,
                "train_acc": run_acc_t,
                "test_loss": run_loss_e,
                "test_acc": run_acc_e,
                #"n_trainable_params": self.model.count_effective_trainable_parameters(),
                "lr": self.config["lr"],
                "n_trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad) + sum(p.numel() for p in self.classifier.parameters() if p.requires_grad),
            }

        return res






# === RNN Trainable class for comparison experiments === #


class RNNTrainable(tune.Trainable):
    def setup(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.build_trial(config)

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict]:
        self.model.to("cpu")
        torch.save(self.model, f"{checkpoint_dir}/model.pth")
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint: Dict):
        self.model = torch.load(checkpoint["model.pth"])
        self.model.to(self.device)


    def build_trial(self, config):
        # Dataset loading (reuse logic from RNNAssemblyTrainable)
        dataset = config.get("dataset", "mnist")
        train_batch_size = config.get("train_batch_size", 64)
        eval_batch_size = config.get("eval_batch_size", 64)
        root = config.get("root", None)
        if dataset == "har2":
            self.train_loader, self.eval_loader, _ = load_har2_dataloaders(
                root=root, train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_har2_input_output_sizes()
        elif dataset == "forda":
            self.train_loader, self.eval_loader, _ = load_forda_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_forda_input_output_sizes()
        elif dataset == "adiac":
            self.train_loader, self.eval_loader, _ = load_adiac_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_adiac_input_output_sizes()
        elif dataset == "fordb":
            self.train_loader, self.eval_loader, _ = load_fordb_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_fordb_input_output_sizes()
        elif dataset == "japanesevowels":
            self.train_loader, self.eval_loader, _ = load_JapaneseVowels_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_JapaneseVowels_input_output_sizes()
        elif dataset == "newstitlesentiment":
            self.train_loader, self.eval_loader, _ = load_NewsTitleSentiment_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_NewsTitleSentiment_input_output_sizes()
        elif dataset == "ieeeppg":
            self.train_loader, self.eval_loader, _ = load_IEEEPPG_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size
            )
            input_size, out_size = get_IEEEPPG_input_output_sizes()
        elif dataset in ["smnist", "psmnist"]:   
            self.train_loader, self.eval_loader = load_mnist(
                train_batch_size=train_batch_size,
                test_batch_size=eval_batch_size,
                permute_seed=config["permute_seed"], 
                root=config["root"]
            )
            input_size = 1
            out_size = 10
        elif dataset == "pems":
            self.train_loader, self.eval_loader, _ = load_PEMS_SF_dataloaders(
                train_batch_size=train_batch_size, test_batch_size=eval_batch_size, 
            )
            input_size, out_size = get_PEMS_SF_input_output_sizes()
        else:
            raise ValueError(f"Dataset {dataset} not supported for RNNTrainable.")

        self.model = nn.RNN(
            input_size=input_size,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("n_layers", 1),
            bidirectional=config.get("bidirectional", False),
            batch_first=False,
            nonlinearity=config.get("nonlinearity", 'tanh'),
        ).to(self.device)
        self.classifier = nn.Linear(
            config.get("hidden_size", 128) * (2 if config.get("bidirectional", False) else 1),
            out_size
        ).to(self.device)
        # Use CrossEntropyLoss for classification, otherwise MSE
        if dataset in regression_datasets:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.opt = Adam(
            list(self.model.parameters()) + list(self.classifier.parameters()),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("l2_reg", 0.0)
        )
        self.lr = MultiStepLR(self.opt, milestones=[80, 160], gamma=0.1)

    def step(self):
        self.model.train()
        self.classifier.train()
        run_loss_t = 0.0
        run_acc_t = 0.0
        n_batches = 0
        self.model.train()
        #""" if you want the progress bar, uncomment the following lines
        from tqdm import tqdm
        for x, y in tqdm(self.train_loader, desc="Batches processed", unit="batch"):
        #"""
        #for x, y in self.train_loader:
            self.opt.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            all_pred, _ = self.model(x)
            
            y_pred = all_pred[-1]
            y_pred = self.classifier(y_pred)
            
            loss = self.loss(y_pred, y)
            loss.backward()
            self.opt.step()

            run_loss_t += loss.item()
            # compute accuracy only for classification tasks
            if self.config.get("dataset", "mnist") not in regression_datasets:
                run_acc_t += (
                    (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item()
                ) / len(y)
            n_batches += 1
        run_loss_t /= n_batches
        # Compute accuracy only for classification tasks
        if self.config.get("dataset", "mnist") not in regression_datasets:
            run_acc_t /= n_batches
        self.lr.step()

        self.model.eval()
        self.classifier.eval()
        run_loss_e = 0.0
        run_acc_e = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                all_pred, _ = self.model(x)
                
                y_pred = all_pred[-1]
                y_pred = self.classifier(y_pred)

                loss = self.loss(y_pred, y)
                run_loss_e += loss.item()
                # compute accuracy only for classification tasks
                if self.config.get("dataset", "mnist") not in regression_datasets:
                    run_acc_e += (
                        (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item()
                    ) / len(y)
                n_batches += 1
        run_loss_e /= n_batches
        # Compute accuracy only for classification tasks
        if self.config.get("dataset", "mnist") not in regression_datasets:
            run_acc_e /= n_batches

        # For regression tasks, log MSE and RMSE instead of accuracy
        if self.config.get("dataset", "mnist") in regression_datasets:
            res = {
                "train_loss": run_loss_t, # this is MSE
                "train_RMSE": math.sqrt(run_loss_t),
                "test_loss": run_loss_e, # this is MSE
                "test_RMSE": math.sqrt(run_loss_e),
                "lr": self.config["lr"],
                "n_trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad) + sum(p.numel() for p in self.classifier.parameters() if p.requires_grad),
            }
        else:
            res = {
                "train_loss": run_loss_t,
                "train_acc": run_acc_t,
                "test_loss": run_loss_e,
                "test_acc": run_acc_e,
                #"n_trainable_params": self.model.count_effective_trainable_parameters(),
                "lr": self.config["lr"],
                "n_trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad) + sum(p.numel() for p in self.classifier.parameters() if p.requires_grad),
            }

        return res

