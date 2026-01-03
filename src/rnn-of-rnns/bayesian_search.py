"""
Bayesian search script for enhanced RNNAssembly experiments.
This script is based on main_search.py but specifically designed for Bayesian optimization with:
1. Focus only on RNNAssembly models (no LSTM/vanilla RNN)
2. Enhanced RNNAssembly with LayerNorm and attention pooling
3. AdamW optimizer with weight decay search
4. Bayesian optimization using Optuna
5. Advanced learning rate scheduling

Added for Bayesian search experiments - does not modify existing codebase.
"""

import os
import argparse
import ray
from ray import tune, train
from ray.tune import CLIReporter, stopper
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

# Import the Bayesian trainable class - added for Bayesian search
from bayesian_trainable import BayesianRNNAssemblyTrainable

# Import the Bayesian search space - added for Bayesian search
from search_space import bayesian_rnnassembly_search_space

from common_utils import (
    load_adiac_dataloaders, load_har2_dataloaders, load_forda_dataloaders, load_fordb_dataloaders,
    load_IEEEPPG_dataloaders, load_NewsTitleSentiment_dataloaders,
    load_JapaneseVowels_dataloaders, 
    load_mnist,
    load_PEMS_SF_dataloaders, 
)


# Dataset configurations for Bayesian search - focused only on RNNAssembly
BAYESIAN_DATASET_CONFIGS = {
    "har2": {
        "loader_fn": load_har2_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "har"),
        "trainable_cls": BayesianRNNAssemblyTrainable,  # Only RNNAssembly for Bayesian search
        "search_space": bayesian_rnnassembly_search_space,
    },
    "forda": {
        "loader_fn": load_forda_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "forda"),
        "trainable_cls": BayesianRNNAssemblyTrainable,
        "search_space": bayesian_rnnassembly_search_space,
    },
    "fordb": {
        "loader_fn": load_fordb_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "fordb"),
        "trainable_cls": BayesianRNNAssemblyTrainable,
        "search_space": bayesian_rnnassembly_search_space,
    },
    "ieeeppg": {
        "loader_fn": load_IEEEPPG_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "ieeeppg"),
        "trainable_cls": BayesianRNNAssemblyTrainable,
        "search_space": bayesian_rnnassembly_search_space,
    },
    "newstitlesentiment": {
        "loader_fn": load_NewsTitleSentiment_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "newstitlesentiment"),
        "trainable_cls": BayesianRNNAssemblyTrainable,
        "search_space": bayesian_rnnassembly_search_space,
    },
    "japanesevowels": {
        "loader_fn": load_JapaneseVowels_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "japanesevowels"),
        "trainable_cls": BayesianRNNAssemblyTrainable,
        "search_space": bayesian_rnnassembly_search_space,
    },
    "adiac": {
        "loader_fn": load_adiac_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "adiac"),
        "trainable_cls": BayesianRNNAssemblyTrainable,
        "search_space": bayesian_rnnassembly_search_space,
    },
    "smnist": {
        "loader_fn": load_mnist,  
        "root": os.path.join(os.getcwd(), "data", "smnist"),
        "trainable_cls": BayesianRNNAssemblyTrainable,
        "search_space": bayesian_rnnassembly_search_space,
    },
    "psmnist": {
        "loader_fn": load_mnist,  
        "root": os.path.join(os.getcwd(), "data", "psmnist"),
        "trainable_cls": BayesianRNNAssemblyTrainable,
        "search_space": bayesian_rnnassembly_search_space,
    },
    "pems": {
        "loader_fn": load_PEMS_SF_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "pems"),
        "trainable_cls": BayesianRNNAssemblyTrainable,
        "search_space": bayesian_rnnassembly_search_space,
    },
}


def main():
    parser = argparse.ArgumentParser()
    # Simplified arguments for Bayesian search - only RNNAssembly
    parser.add_argument("--datasets", nargs="+", default=["fordb"], 
                       help="Datasets to run Bayesian search on")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs to use")
    parser.add_argument("--num_gpus", type=float, default=0.5, help="Number of GPUs per trial")
    parser.add_argument("--num_cpus", type=int, default=8, help="Number of CPUs per trial")
    parser.add_argument("--results_dir", type=str, default="/storagenfs/a052721/AdaDiag/bayesian_search/", 
                       help="Directory to save Bayesian search results")
    parser.add_argument("--ray_tmp", type=str, default="/storagenfs/a052721/ray_tmp", 
                       help="Ray temp directory")
    parser.add_argument("--checkpoint", action="store_true", help="Enable model checkpointing")
    parser.add_argument("--checkpoint_metric", type=str, default="test_acc", 
                       help="Metric for checkpoint selection")
    parser.add_argument("--checkpoint_order", type=str, default="max", choices=["max", "min"], 
                       help="Order for checkpoint metric")
    parser.add_argument("--checkpoint_freq", type=int, default=1, help="Checkpoint frequency (epochs)")
    
    # Bayesian search specific arguments
    parser.add_argument("--n_trials", type=int, default=100, 
                       help="Number of Bayesian optimization trials")
    parser.add_argument("--study_name", type=str, default="bayesian_rnnassembly", 
                       help="Optuna study name")
    
    # Override arguments for Bayesian search
    parser.add_argument("--train_batch_size", type=int, default=None, 
                       help="Override train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=None, 
                       help="Override eval batch size")
    parser.add_argument("--max_epochs", type=int, default=None, 
                       help="Override max epochs")
    

    # Example of usage:
    # python bayesian_search.py --datasets adiac --gpus 0 --n_trials 150
    
    args = parser.parse_args()

    # Set up Ray environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["RAY_TMPDIR"] = args.ray_tmp
    os.environ["RAY_SESSION_DIR"] = args.ray_tmp
    os.makedirs(args.ray_tmp, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    ray.init()
    
    for dataset_name in args.datasets:
        cfg = BAYESIAN_DATASET_CONFIGS[dataset_name]
        trainable_cls = cfg["trainable_cls"]
        search_space_fn = cfg["search_space"]
        
        # Get search space for this dataset
        search_space = search_space_fn(dataset_name)
        search_space["dataset"] = dataset_name
        search_space["root"] = cfg["root"]
        
        # Override parameters if provided
        if args.train_batch_size is not None:
            search_space["train_batch_size"] = args.train_batch_size
        if args.eval_batch_size is not None:
            search_space["eval_batch_size"] = args.eval_batch_size
        if args.max_epochs is not None:
            search_space["max_epochs"] = args.max_epochs
        if args.n_trials is not None:
            search_space["n_trials"] = args.n_trials
            
        # Ensure permute_seed is always present
        if "permute_seed" not in search_space:
            if dataset_name == "psmnist":
                search_space["permute_seed"] = 0
            else:
                search_space["permute_seed"] = None

        n_trials = search_space.get("n_trials", 100)

        # Structured results directory per dataset
        dataset_results_dir = os.path.join(args.results_dir, dataset_name, "enhanced_rnnassembly")
        os.makedirs(dataset_results_dir, exist_ok=True)

        # Enhanced reporter for Bayesian search
        reporter = CLIReporter(
            parameter_columns=[
                "dataset", "block_sizes", "activation", "lr", "weight_decay", 
                "gate_bounds", "coupling_topology", "lr_scheduler",
                "use_layer_norm", "use_attention_pooling"
            ],
            metric_columns=["train_loss", "train_acc", "test_loss", "test_acc", "n_trainable_params"],
        )

        checkpoint_config = None
        if args.checkpoint:
            checkpoint_config = train.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=args.checkpoint_metric,
                checkpoint_score_order=args.checkpoint_order,
                checkpoint_frequency=args.checkpoint_freq,
            )

        # Bayesian optimization setup - added for Bayesian search
        search_alg = OptunaSearch(
            metric="test_acc" if dataset_name not in ["ieeeppg", "newstitlesentiment"] else "test_loss",
            mode="max" if dataset_name not in ["ieeeppg", "newstitlesentiment"] else "min",
        )
        
        # Early stopping scheduler for efficiency - added for Bayesian search
        # Get a concrete max_epochs value for the scheduler
        max_epochs_value = 200  # Default value
        if "max_epochs" in search_space:
            max_epochs_param = search_space["max_epochs"]
            if hasattr(max_epochs_param, 'categories'):
                # If it's a Categorical, use the maximum value
                max_epochs_value = max(max_epochs_param.categories)
            elif isinstance(max_epochs_param, (int, float)):
                max_epochs_value = int(max_epochs_param)
        
        scheduler = ASHAScheduler(
            metric="test_acc" if dataset_name not in ["ieeeppg", "newstitlesentiment"] else "test_loss",
            mode="max" if dataset_name not in ["ieeeppg", "newstitlesentiment"] else "min",
            max_t=max_epochs_value,
            grace_period=30,
            reduction_factor=2,
        )

        run_config = train.RunConfig(
            storage_path=dataset_results_dir,
            checkpoint_config=checkpoint_config,
            stop=stopper.MaximumIterationStopper(max_epochs_value),
            progress_reporter=reporter,
        )

        # Bayesian optimization tune config
        tune_config = tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=n_trials
        )

        tuner = tune.Tuner(
            tune.with_resources(trainable_cls, {"cpu": args.num_cpus, "gpu": args.num_gpus}),
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
        )

        print(f"Starting Bayesian search for {dataset_name} (Enhanced RNNAssembly)...")
        print(f"Search space includes:")
        print(f"  - AdamW weight decay: 1e-6 to 1e-3")
        print(f"  - Learning rate: 1e-4 to 1e-1") 
        print(f"  - Gate bounds: a(1e-6 to 1e-2), b(1e-2 to 1.0)")
        print(f"  - Coupling topology: 2 to 1000")
        print(f"  - Block sizes: A(2-512) for _ in range(B(2-128))")
        print(f"  - LR schedulers: cosine, onecycle, plateau")
        print(f"  - LayerNorm and attention pooling options")
        print(f"  - Number of trials: {n_trials}")
        
        tuner.fit()
        print(f"Completed Bayesian search for {dataset_name}. Results in {dataset_results_dir}")


if __name__ == "__main__":
    main()