import os
import argparse
import ray
from ray import tune, train
from ray.tune import CLIReporter, stopper
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from trainable import RNNAssemblyTrainable

from search_space import (
    AdaDiag_extended_search_space,
)

from common_utils import (
    load_adiac_dataloaders, load_har2_dataloaders, load_forda_dataloaders, load_fordb_dataloaders,
    load_IEEEPPG_dataloaders, load_NewsTitleSentiment_dataloaders, load_FloodModeling1_dataloaders,
    load_JapaneseVowels_dataloaders, 
    load_mnist,
    load_PEMS_SF_dataloaders, 
)


DATASET_CONFIGS = {
    "har2": {
        "loader_fn": load_har2_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "har"),
        "trainable_cls": {
            "rnnassembly": RNNAssemblyTrainable,
            "scr": RNNAssemblyTrainable,
        },
        "search_spaces": {
            "rnnassembly": AdaDiag_extended_search_space,
        },
    },
    "forda": {
        "loader_fn": load_forda_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "forda"),
        "trainable_cls": {
            "rnnassembly": RNNAssemblyTrainable,
            "scr": RNNAssemblyTrainable,
        },
        "search_spaces": {
            #"rnnassembly": AdaDiag_search_space,
            "rnnassembly": AdaDiag_extended_search_space,
        },
    },
    "fordb": {
        "loader_fn": load_fordb_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "fordb"),
        "trainable_cls": {
            "rnnassembly": RNNAssemblyTrainable,
            "scr": RNNAssemblyTrainable,
        },
        "search_spaces": {
            "rnnassembly": AdaDiag_extended_search_space,
        },
    },
    "ieeeppg": {
        "loader_fn": load_IEEEPPG_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "ieeeppg"),
        "trainable_cls": {
            "rnnassembly": RNNAssemblyTrainable,
            "scr": RNNAssemblyTrainable,
        },
        "search_spaces": {
            "rnnassembly": AdaDiag_extended_search_space,
        },
    },
    "newstitlesentiment": {
        "loader_fn": load_NewsTitleSentiment_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "newstitlesentiment"),
        "trainable_cls": {
            "rnnassembly": RNNAssemblyTrainable,
            "scr": RNNAssemblyTrainable,
        },
        "search_spaces": {
            "rnnassembly": AdaDiag_extended_search_space,
        },
    },
    "japanesevowels": {
        "loader_fn": load_JapaneseVowels_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "japanesevowels"),
        "trainable_cls": {
            "rnnassembly": RNNAssemblyTrainable,
            "scr": RNNAssemblyTrainable,
        },
        "search_spaces": {
            "rnnassembly": AdaDiag_extended_search_space,
        },
    },
    "adiac": {
        "loader_fn": load_adiac_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "adiac"),
        "trainable_cls": {
            "rnnassembly": RNNAssemblyTrainable,
            "scr": RNNAssemblyTrainable,
        },
        "search_spaces": {
            "rnnassembly": AdaDiag_extended_search_space,
        },
    },
    "smnist": {
        "loader_fn": load_mnist,  
        "root": os.path.join(os.getcwd(), "data", "smnist"),
        "trainable_cls": {
            "rnnassembly": RNNAssemblyTrainable,
            "scr": RNNAssemblyTrainable,
        },
        "search_spaces": {
            "rnnassembly": AdaDiag_extended_search_space,
        },
    },
    "psmnist": {
        "loader_fn": load_mnist,  
        "root": os.path.join(os.getcwd(), "data", "psmnist"),
        "trainable_cls": {
            "rnnassembly": RNNAssemblyTrainable,
            "scr": RNNAssemblyTrainable,
        },
        "search_spaces": {
            "rnnassembly": AdaDiag_extended_search_space,
        },
    },
    "pems": {
        "loader_fn": load_PEMS_SF_dataloaders,
        "root": os.path.join(os.getcwd(), "data", "pems"),
        "trainable_cls": {      
            "rnnassembly": RNNAssemblyTrainable,
            "scr": RNNAssemblyTrainable,
        },
        "search_spaces": {
            "rnnassembly": AdaDiag_extended_search_space,
        },
    },
}

def trainable(config):
    # Deprecated: use Trainable classes directly
    #dataset = config["dataset"]
    #loader_fn = DATASET_CONFIGS[dataset]["loader_fn"]
    #root = DATASET_CONFIGS[dataset]["root"]
    #train_loader, val_loader, test_loader = loader_fn(
    #    config.get("train_batch_size", 64),
    #    config.get("eval_batch_size", 64)
    #)
    # Model training logic goes here
    #tune.report(metric=0.0)
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", nargs="+", default=["rnnassembly"], help="Model types to use for search")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()), help="Datasets to run grid search on")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs to use")
    parser.add_argument("--num_gpus", type=float, default=0.5, help="Number of GPUs per trial")
    parser.add_argument("--num_cpus", type=int, default=8, help="Number of CPUs per trial")
    parser.add_argument("--results_dir", type=str, default="/storagenfs/a052721/AdaDiag/extended_search/", help="Directory to save results")
    parser.add_argument("--ray_tmp", type=str, default="/storagenfs/a052721/ray_tmp", help="Ray temp directory")
    parser.add_argument("--checkpoint", action="store_true", help="Enable model checkpointing")
    parser.add_argument("--checkpoint_metric", type=str, default="test_acc", help="Metric for checkpoint selection")
    parser.add_argument("--checkpoint_order", type=str, default="max", choices=["max", "min"], help="Order for checkpoint metric")
    parser.add_argument("--checkpoint_freq", type=int, default=1, help="Checkpoint frequency (epochs)")
    parser.add_argument("--train_batch_size", type=int, default=None, help="Override the default train batch size defined in search_space")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Override the default eval batch size defined in search_space")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override the default max_epochs defined in search_space")
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=None, help="Override the default decay_epochs list defined in search_space")
    parser.add_argument("--ring", action="store_true", help="Force ring topology for rnnassembly")
    parser.add_argument("--store_frequencies", action="store_true", help="Store gated_step tensor for each time step and epoch")
    args = parser.parse_args()

    # Final search commands:
    # Completed:
    #python main_search.py --datasets japanesevowels adiac har2 --gpus 2 --num_gpus 0.1 --num_cpus 8 --model_type scr lstm rnn rnnassembly
    #python main_search.py --datasets newstitlesentiment --train_batch_size 1024 --eval_batch_size 1024 --max_epochs 10 --decay_epochs [5,8] --gpus 2 --num_gpus 0.5 --num_cpus 8 --model_type lstm scr rnn rnnassembly    
    #python main_search.py --datasets ieeeppg --max_epochs 50 --decay_epochs 20 40 --gpus 3 --num_gpus 0.5 --num_cpus 8 --model_type scr
    #stopped(python main_search.py --datasets ieeeppg --max_epochs 250 --decay_epochs 80 160 --gpus 3 --num_gpus 0.5 --num_cpus 8 --model_type lstm
    #python main_search.py --datasets ieeeppg --max_epochs 50 --decay_epochs 20 40 --gpus 3 --num_gpus 0.5 --num_cpus 8 --model_type rnnassembly    
    #python main_search.py --datasets ieeeppg --max_epochs 250 --decay_epochs 80 160 --gpus 3 --num_gpus 0.5 --num_cpus 8 --model_type rnn    
    #stopped(python main_search.py --datasets emopain facedetection heartbeat motionsensehar --gpus 3 --num_gpus 0.25 --num_cpus 8 --model_type scr rnnassembly lstm rnn)   
    #python main_search.py --datasets heartbeat motionsensehar emopain facedetection--gpus 3 --num_gpus 0.25 --num_cpus 8 --model_type scr rnnassembly
    #python main_search.py --datasets forda blink fordb --gpus 2 --num_gpus 0.25 --num_cpus 8 --model_type lstm scr rnnassembly rnn    
    #python main_search.py --datasets motionsensehar --gpus 3 --num_gpus 1. --num_cpus 8 --model_type rnnassembly scr
    #python main_search.py --datasets pems --gpus 2 --num_gpus 1.0 --num_cpus 8 --model_type rnnassembly scr
    #python main_search.py --datasets smnist psmnist --train_batch_size 512 --eval_batch_size 512 --gpus 0,1,2,3 --num_gpus 0.5 --num_cpus 8 --model_type rnnassembly scr

    ###################################################################################################################################
    # Extended search with cosine annealing LR scheduler:
    # python extended_search.py --datasets forda fordb --gpus 3 --num_gpus 1 --num_cpus 1 --model_type rnnassembly
    ###################################################################################################################################

    # ##### Workaround for Ray since /tmp is over 95% full #####
    #custom_ray_dir = "/storagenfs/a052721/ray_tmp" # Change this to a directory with sufficient space

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # Set the environment variables for Ray to use this directory
    os.environ["RAY_TMPDIR"] = args.ray_tmp
    os.environ["RAY_SESSION_DIR"] = args.ray_tmp
    os.makedirs(args.ray_tmp, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    ray.init()
    for dataset_name in args.datasets:
        cfg = DATASET_CONFIGS[dataset_name]
        model_types = getattr(args, "model_type", ["rnnassembly"])
        for model_type in model_types:
            trainable_cls = cfg.get("trainable_cls", {}).get(model_type, RNNAssemblyTrainable)
            search_space_fn = cfg["search_spaces"][model_type]
            # Pass dataset_name if required
            try:
                search_space = search_space_fn(dataset_name)
            except TypeError:
                search_space = search_space_fn()
            search_space["dataset"] = dataset_name
            search_space["root"] = cfg["root"]
            # Overwrite batch sizes and epochs if provided by user
            if args.train_batch_size is not None:
                search_space["train_batch_size"] = args.train_batch_size
            if args.eval_batch_size is not None:
                search_space["eval_batch_size"] = args.eval_batch_size
            if args.max_epochs is not None:
                search_space["max_epochs"] = args.max_epochs
            if args.decay_epochs is not None:
                search_space["decay_epochs"] = args.decay_epochs
            if args.ring:
                if model_type == "rnnassembly":
                    search_space["block_structure"] = "ring"
                    print("Forcing the ring topology for rnnassembly blocks.")
                else:
                    print("WARNING. Ring topology is only applicable to rnnassembly model type. Ignoring --ring flag.")
            ############################################
            # added for storing frequencies
            # Pass store_frequencies flag from CLI to search_space/config
            if getattr(args, "store_frequencies", False):
                search_space["store_frequencies"] = True
            ############################################
            # Ensure permute_seed is always present
            if "permute_seed" not in search_space:
                if dataset_name == "psmnist":
                    search_space["permute_seed"] = 0
                else:
                    search_space["permute_seed"] = None
            n_trials = search_space.get("n_trials", 1)

            # Structured results directory per dataset and model_type
            dataset_results_dir = os.path.join(args.results_dir, dataset_name, model_type)
            os.makedirs(dataset_results_dir, exist_ok=True)

            reporter = CLIReporter(
                parameter_columns=["dataset", "block_config", "activation", "eul_step", "gating", "coupling_topology"],
                metric_columns=["train_loss", "train_acc", "test_loss", "test_acc"],
            )

            checkpoint_config = None
            if args.checkpoint:
                checkpoint_config = train.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute=args.checkpoint_metric,
                    checkpoint_score_order=args.checkpoint_order,
                    checkpoint_frequency=args.checkpoint_freq,
                )

            run_config = train.RunConfig(
                storage_path=dataset_results_dir,
                checkpoint_config=checkpoint_config,
                stop=stopper.MaximumIterationStopper(search_space.get("max_epochs", 200)),
                progress_reporter=reporter,
            )

            tune_config = tune.TuneConfig(num_samples=n_trials)

            tuner = tune.Tuner(
                tune.with_resources(trainable_cls, {"cpu": args.num_cpus, "gpu": args.num_gpus}),
                param_space=search_space,
                tune_config=tune_config,
                run_config=run_config,
            )

            print(f"Starting grid search for {dataset_name} ({model_type})...")
            tuner.fit()
            print(f"Completed grid search for {dataset_name} ({model_type}). Results in {dataset_results_dir}")

if __name__ == "__main__":
    main()