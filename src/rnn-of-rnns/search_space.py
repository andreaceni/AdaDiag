import os
import torch
from ray import tune



#################################
# Search for different models
#################################

def AdaDiag_search_space(dataset_name):
    """
    Search space configuration for AdaDiag experiments.
    """
    import torch
    from ray import tune
    permute_seed = None
    if dataset_name == "psmnist":
        permute_seed = 0
    elif dataset_name == "smnist":
        permute_seed = None
    return {
        #"root": os.path.join(os.getcwd(), "data", "har"),  # HAR dataset path TODO: change path
        "dataset": dataset_name,
        "permute_seed": permute_seed,
        "train_batch_size": tune.grid_search([128]),
        "eval_batch_size": tune.grid_search([128]),
        "input_size": 1,
        "out_size": 2,
        "block_sizes": tune.grid_search([
            [32 for _ in range(16)],  
            [8 for _ in range(64)],
        ]),
        "block_config": tune.grid_search([3,13]),
        "coupling_block_init_fn": ("uniform",),
        "coupling_topology": tune.grid_search([ 5, 20, 500]),
        "eul_step": 0.1,
        "gamma": None,
        "activation": tune.grid_search(["relu"]),
        "dtype": torch.float32,
        "decay_epochs": [80, 160],
        "decay_scalar": 0.1,
        #"ortho_option": tune.grid_search(["permutation"]), 
        "check_stability": True,
        "max_epochs": 250,
        "n_trials": 1,
        "gating": tune.grid_search([True]),
        "gate_bounds": tune.grid_search([(0.00001, 0.2)]),
        "coupling_rescaling": "unconstrained",
        "initialisation": "default",
        "block_structure": tune.grid_search(["identity"]),
        "model_mode": tune.grid_search(["standard"]),
        "lr": tune.grid_search([1e-3, 1e-2]), # learning rate
    }


def SCR_search_space(dataset_name):
    """
    Search space configuration for SCR experiments.
    """
    import torch
    from ray import tune
    return {
        "dataset": dataset_name,
        "train_batch_size": tune.grid_search([128]),
        "eval_batch_size": tune.grid_search([128]),
        "block_sizes": tune.grid_search([
            [32 for _ in range(16)],  
            [8 for _ in range(64)],
        ]),
        "block_config": tune.grid_search([2,1]),
        "coupling_block_init_fn": ("uniform",),
        "coupling_topology": tune.grid_search([ 20, 5, 500]),
        "eul_step": tune.grid_search([0.01, 0.1, 0.001]),
        "gamma": None,
        "activation": tune.grid_search(["relu"]),
        "dtype": torch.float32,
        "decay_epochs": [80, 160],
        "decay_scalar": 0.1,
        "check_stability": True,
        "max_epochs": 250,
        "n_trials": 1,
        "gating": tune.grid_search([False]),
        "gate_bounds": (0.,0.),
        "coupling_rescaling": "unconstrained",
        "initialisation": "default",
        "lr": tune.grid_search([1e-3, 1e-2]), # learning rate
    }

def lstm_search_space(dataset_name):
    from ray import tune
    return {
        #"root": os.path.join(os.getcwd(), "data", "har"),
        "dataset": dataset_name,
        "train_batch_size": tune.grid_search([128]),
        "eval_batch_size": tune.grid_search([128]),
        "dtype": torch.float32,
        "max_epochs": 250,
        "n_trials": 1,
        "decay_epochs": [80, 160],
        "decay_scalar": 0.1,
        "hidden_size": tune.grid_search([64,128]),
        "n_layers": tune.grid_search([3,2,1]),
        "bidirectional": tune.grid_search([True, False]),
        "lr": tune.grid_search([1e-2, 1e-3]),
        "l2_reg": tune.grid_search([0.0, 1e-4]),
    }

def rnn_search_space(dataset_name):
    from ray import tune
    return {
        #"root": os.path.join(os.getcwd(), "data", "har"),
        "dataset": dataset_name,
        "train_batch_size": tune.grid_search([128]),
        "eval_batch_size": tune.grid_search([128]),
        "dtype": torch.float32,
        "max_epochs": 250,
        "n_trials": 1,
        "decay_epochs": [80, 160],
        "decay_scalar": 0.1,
        "hidden_size": tune.grid_search([64,128]),
        "n_layers": tune.grid_search([3,2,1]),
        "bidirectional": tune.grid_search([True, False]),
        "lr": tune.grid_search([1e-2, 1e-3]),
        "l2_reg": tune.grid_search([0.0, 1e-4]),
    }

# =============================================================


def AdaDiag_for_plot_withoutSkewSymmetry(dataset_name):
    """
    Configuration for AdaDiag plot without Skew Symmetry.
    """
    import torch
    from ray import tune
    permute_seed = None
    if dataset_name == "psmnist":
        permute_seed = 0
    elif dataset_name == "smnist":
        permute_seed = None
    return {
        #"root": os.path.join(os.getcwd(), "data", "har"),  # HAR dataset path TODO: change path
        "dataset": dataset_name,
        "permute_seed": permute_seed,
        "train_batch_size": 128,
        "eval_batch_size": 128,
        "input_size": 1,
        "out_size": 2,
        "block_sizes": [32 for _ in range(16)],
        #"block_sizes": [8 for _ in range(64)],
        "block_config": tune.grid_search([3]),
        "coupling_block_init_fn": ("uniform",),
        "coupling_topology": 20,
        "eul_step": 0.1,
        "gamma": None,
        "activation": "relu",
        "dtype": torch.float32,
        "decay_epochs": [80, 160],
        "decay_scalar": 0.1,
        #"ortho_option": "permutation", 
        "check_stability": True,
        "max_epochs": 250,
        "n_trials": 1,
        "gating": True,
        "gate_bounds": (0.00001, 0.2),
        "coupling_rescaling": "unconstrained",
        "initialisation": "default",
        "block_structure": "identity",
        "model_mode": "standard",
        "lr": 1e-2, # learning rate
    }


def AdaDiag_for_plot_Frequencies(dataset_name):
    """
    Configuration for AdaDiag plot Frequencies.
    """
    import torch
    from ray import tune
    permute_seed = None
    if dataset_name == "psmnist":
        permute_seed = 0
    elif dataset_name == "smnist":
        permute_seed = None
    return {
        #"root": os.path.join(os.getcwd(), "data", "har"),  # HAR dataset path TODO: change path
        "dataset": dataset_name,
        "permute_seed": permute_seed,
        "train_batch_size": 128,
        "eval_batch_size": 128,
        "input_size": 1,
        "out_size": 2,
        "block_sizes": [32 for _ in range(16)],
        #"block_sizes": [8 for _ in range(64)],
        "block_config": tune.grid_search([3]),
        "coupling_block_init_fn": ("uniform",),
        "coupling_topology": 5,
        "eul_step": 0.1,
        "gamma": None,
        "activation": "relu",
        "dtype": torch.float32,
        "decay_epochs": [80, 160],
        "decay_scalar": 0.1,
        #"ortho_option": "permutation", 
        "check_stability": True,
        "max_epochs": 250,
        "n_trials": 1,
        "gating": True,
        "gate_bounds": (0.00001, 0.2),
        "coupling_rescaling": "unconstrained",
        "initialisation": "default",
        "block_structure": "identity",
        "model_mode": "standard",
        "lr": 1e-2, # learning rate
    }
