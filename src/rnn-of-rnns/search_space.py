import os
import torch
from ray import tune


"""
def search_space():
    return {
        "root": os.path.join(os.getcwd(), "data"),
        "permute_seed": None,
        "input_size": 1,
        "out_size": 10,
        "block_sizes": [32 for _ in range(16)],
        "block_config": tune.grid_search([5, 6]),  # 5: diag_norm (lt), 6: diag_norm (diagonal), 7: proper_diag_norm
        "coupling_block_init_fn": ("uniform",),
        "coupling_topology": tune.grid_search([
            20, 0.3, 0.8,
            {"exact_row_couplings": 5},
            {"exact_row_couplings": 20}
        ]),  # Can also be an explicit list of pairs, e.g. [(0, 1), (2, 3)], or a dict {"exact_row_couplings": N} to specify N nonzero off-diagonal blocks per block row.
        "eul_step": tune.grid_search([0.03, 0.01, 0.001, 0.005]),
        "gamma": None, 
        "activation": "tanh",  # If dtype is complex, the activation is always cardioid regardless of this value.
        "dtype": tune.grid_search([torch.complex64, torch.float32]),
        "decay_epochs": [80, 150],
        "decay_scalar": 0.1,
        "check_stability": False,  # Set to True to enable spectral radius/norm logging
    }
"""



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
            [32 for _ in range(16)],  # Standard configuration
            [8 for _ in range(64)],
        ]),
        #"block_config": tune.grid_search([3]),
        "block_config": tune.grid_search([13]),
        "coupling_block_init_fn": ("uniform",),
        "coupling_topology": tune.grid_search([ 5, 20, 500]),
        #"coupling_topology": tune.grid_search([ 60 ]),
        "eul_step": 0.1,
        "gamma": None,
        "activation": tune.grid_search(["relu"]),
        "dtype": torch.float32,
        "decay_epochs": [80, 160],
        "decay_scalar": 0.1,
        "ortho_option": tune.grid_search(["permutation"]), # "unitary" (default) or "permutation", only for idx=8 (i.e. ortho_diag blocks)
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
            [32 for _ in range(16)],  # Standard configuration
            [8 for _ in range(64)],
        ]),
        "block_config": tune.grid_search([2,1]),
        "coupling_block_init_fn": ("uniform",),
        "coupling_topology": tune.grid_search([ 20, 5, 500]),
        #"coupling_topology": tune.grid_search([ 60]),
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
        "ortho_option": "permutation", # "unitary" (default) or "permutation", only for idx=8 (i.e. ortho_diag blocks)
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
        "ortho_option": "permutation", # "unitary" (default) or "permutation", only for idx=8 (i.e. ortho_diag blocks)
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


# =============================================================



import math
import random

def loguniform(low, high):
    """Sample a log-uniform value."""
    log_low = math.log(low)
    log_high = math.log(high)
    return math.exp(random.random() * (log_high - log_low) + log_low)


def AdaDiag_extended_search_space(dataset_name):
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
        "train_batch_size": 128, # 128 for fordA, fordB and MNIST-based, 16 for Adiac.
        "eval_batch_size": 128,
        "block_sizes": tune.sample_from(
            lambda _: (
                lambda A, B: [A for _ in range(B)]  # <------- random search on A and B where [A for _ in range B]; A=loguniform(2,512), B=128-loguniform(2,128)
            )(
                A = int(loguniform(2, 512)),           # A sampled here
                B = int(128 - loguniform(2, 128)) # B sampled here (your definition)
            )
        ),
        "block_config": 3,
        "coupling_block_init_fn": ("uniform",),
        "coupling_topology": tune.qloguniform(2, 1000, q=1), # <-------------- random search here with loguniform distr tune.loguniform(2, 1000)
        "eul_step": 0.1,
        #"gamma": None,
        "activation": "relu",
        "dtype": torch.float32,
        "decay_epochs": [80, 160], # <------ useless since we now use CosineAnnealing LR scheduler
        "decay_scalar": 0.1, # <------ useless since we now use CosineAnnealing LR scheduler
        #"ortho_option": tune.grid_search(["permutation"]), # "unitary" (default) or "permutation", only for idx=8 (i.e. ortho_diag blocks)
        "check_stability": True,
        "max_epochs": 300,
        "n_trials": 500, # <----------------------------- THIS DEFINES HOW MANY RANDOM TRIALS
        "gating": True,
        "gate_bounds": tune.sample_from(  # <------------ random search here on (a,b) with a=loguniform(1e-6,1e-2), b=loguniform(1e-2, 1.)
            lambda _: (
                loguniform(1e-6, 1e-2),   # a
                loguniform(1e-2, 1.0)     # b
            )
        ),
        "coupling_rescaling": "unconstrained",
        "initialisation": "default",
        "block_structure": "identity",
        "model_mode": "standard",
        "lr": tune.loguniform(1e-4, 1e-1), # learning rate <----------------- random search here on gaussian around 0.005
    }







# =============================================================
# Bayesian search space for enhanced RNNAssembly experiments
# Added for Bayesian search experiments
# =============================================================

import math
import random

def loguniform_bayesian(low, high):
    """Sample a log-uniform value for Bayesian search."""
    log_low = math.log(low)
    log_high = math.log(high)
    return math.exp(random.random() * (log_high - log_low) + log_low)


def bayesian_rnnassembly_search_space(dataset_name):
    """
    Bayesian search space configuration for enhanced RNNAssembly experiments.
    
    This search space is specifically designed for Bayesian optimization and includes:
    1. Weight decay parameter for AdamW optimizer (0 to 1e-3)
    2. Learning rate optimization (1e-4 to 1e-1)
    3. Gate bounds optimization (a: 1e-6 to 1e-2, b: 1e-2 to 1.0)
    4. Coupling topology optimization (2 to 1000)
    5. Block sizes optimization ([A for _ in range(B)] where A: 2-512, B: 2-128)
    6. Learning rate scheduling options
    
    Added for Bayesian search experiments.
    """
    import torch
    from ray import tune
    
    permute_seed = None
    if dataset_name == "psmnist":
        permute_seed = 0
    elif dataset_name == "smnist":
        permute_seed = None
    
    return {
        "dataset": dataset_name,
        "permute_seed": permute_seed,
        "train_batch_size": 128, #tune.choice([64, 128]),
        "eval_batch_size": 128, #tune.choice([64, 128]),
        
        # Block sizes: [A for _ in range(B)] where A: 2-512, B: 2-64 (reduced for memory)
        "block_sizes": tune.sample_from(
            lambda spec: [
                int(loguniform_bayesian(2, 256))  # Reduced max block size for memory
                for _ in range(int(loguniform_bayesian(2, 64)))  # Loguniform B parameter
            ]
        ),
        
        # Use only block_config 3 for enhanced experiments
        "block_config": 3,
        "coupling_block_init_fn": ("uniform",),
        
        # Coupling topology: integer between 2 and 1000
        "coupling_topology": tune.qloguniform(2, 1000, q=1),
        
        "eul_step": 0.1,
        "activation": "relu",
        "dtype": torch.float32,
        "decay_epochs": [80, 160],
        "decay_scalar": 0.1,
        "check_stability": True,
        "max_epochs": tune.choice([200, 250, 300]),
        "n_trials": 100,  # Number of Bayesian optimization trials
        
        # Gating parameters
        "gating": True,
        # Gate bounds: (a, b) where a: 1e-6 to 1e-2, b: 1e-2 to 1.0
        "gate_bounds": tune.sample_from(
            lambda spec: (
                loguniform_bayesian(1e-6, 1e-2),
                loguniform_bayesian(1e-2, 1.0)
            )
        ),
        
        "coupling_rescaling": "unconstrained",
        "initialisation": "default",
        "block_structure": "identity",
        "model_mode": "standard",
        
        # Learning rate: between 1e-4 and 1e-1
        "lr": tune.loguniform(1e-4, 1e-1),
        
        # AdamW weight decay: between 0 and 1e-3
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        
        # Learning rate scheduler options
        "lr_scheduler": tune.choice(["cosine", "onecycle", "plateau"]),
        
        # Enhanced RNNAssembly features
        "use_layer_norm": tune.choice([True, False]),
        "use_attention_pooling": tune.choice([True, False]),
    }
