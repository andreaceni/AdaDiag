import os
import argparse
import torch
import numpy as np
from torch import nn
from common_utils import (
    load_har2_dataloaders, get_har2_input_output_sizes,
    load_forda_dataloaders, get_forda_input_output_sizes,
    load_adiac_dataloaders, get_adiac_input_output_sizes,
    load_JapaneseVowels_dataloaders, get_JapaneseVowels_input_output_sizes,
    load_NewsTitleSentiment_dataloaders, get_NewsTitleSentiment_input_output_sizes,
    load_IEEEPPG_dataloaders, get_IEEEPPG_input_output_sizes,
    load_PEMS_SF_dataloaders, get_PEMS_SF_input_output_sizes,
    load_fordb_dataloaders, get_fordb_input_output_sizes,
    load_mnist, get_mnist_input_output_sizes,
    seq_collate_fn
)
from torchdyno.models.rnn_assembly import RNNAssembly
from torchdyno.models.initializers import (
    uniform, normal, orthogonal, sparse, ones, zeros, diagonal, ring, lower_feedforward, lower_triangular
)

DATASET_LOADERS = {
    "har2": (load_har2_dataloaders, get_har2_input_output_sizes),
    "forda": (load_forda_dataloaders, get_forda_input_output_sizes),
    "adiac": (load_adiac_dataloaders, get_adiac_input_output_sizes),
    "japanesevowels": (load_JapaneseVowels_dataloaders, get_JapaneseVowels_input_output_sizes),
    "newstitlesentiment": (load_NewsTitleSentiment_dataloaders, get_NewsTitleSentiment_input_output_sizes),
    "ieeeppg": (load_IEEEPPG_dataloaders, get_IEEEPPG_input_output_sizes),
    "pems": (load_PEMS_SF_dataloaders, get_PEMS_SF_input_output_sizes),
    "fordb": (load_fordb_dataloaders, get_fordb_input_output_sizes),
    "smnist": (load_mnist, get_mnist_input_output_sizes),
    "psmnist": (load_mnist, get_mnist_input_output_sizes),
}

def train(model, train_loader, test_loader, device, 
          classifier=None,
          epochs=10, is_regression=False,
          lr=1e-3, wd=0.0, decay_epochs=[80, 160], decay_scalar=0.1, log_file=None, epoch_best_scores=None):
    model.train()
    classifier.train() if classifier is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=decay_scalar)
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        from tqdm import tqdm
        for x, y in tqdm(train_loader, desc="Batches processed", unit="batch"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            all_pred, _ = model(x)
            if is_regression and classifier is None:
                y_pred = all_pred[-1].view(-1)
            else:
                y_pred = all_pred[-1]
            # deal with classifier for LSTM/RNN
            if classifier is not None:
                y_pred = classifier(y_pred)
                if is_regression:
                    y_pred = y_pred.view(-1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if not is_regression:
                total_acc += (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item() / len(y)
            n_batches += 1
        scheduler.step()
        train_loss = total_loss/n_batches
        train_acc = total_acc/n_batches if not is_regression else None

        # Evaluate on test set at each epoch
        was_training = model.training
        model.eval()
        classifier.eval() if classifier is not None else None
        test_loss = 0.0
        test_acc = 0.0
        test_batches = 0
        preds = []
        targets = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                all_pred, _ = model(x)
                if is_regression and classifier is None:
                    y_pred = all_pred[-1].view(-1)
                    preds.append(y_pred.cpu().numpy())
                    targets.append(y.cpu().numpy())
                else:
                    y_pred = all_pred[-1]
                # deal with classifier for LSTM/RNN
                if classifier is not None:
                    y_pred = classifier(y_pred)
                    if is_regression:
                        y_pred = y_pred.view(-1)
                        preds.append(y_pred.cpu().numpy())
                        targets.append(y.cpu().numpy())
                loss = criterion(y_pred, y)
                test_loss += loss.item()
                if is_regression:
                    pass
                else:
                    test_acc += (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item() / len(y)
                test_batches += 1
        test_loss /= test_batches
        if is_regression:
            preds = np.concatenate(preds)
            targets = np.concatenate(targets)
            test_metric = np.sqrt(np.mean((preds - targets) ** 2))
            metric_name = "Test RMSE"
            score_for_best = test_metric
        else:
            test_acc /= test_batches
            test_metric = test_acc
            metric_name = "Test Accuracy"
            score_for_best = test_metric
        if was_training:
            model.train() # Return to training mode if was training
            classifier.train() if classifier is not None else None

        if epoch_best_scores is not None:
            epoch_best_scores.append(score_for_best)

        # Check lr scheduler
        #current_lr = scheduler.optimizer.param_groups[0]["lr"]
        epoch_log = f"Epoch {epoch+1}: TrainLoss={train_loss:.4f}"
        if not is_regression:
            epoch_log += f", TrainAcc={train_acc:.4f}"
        epoch_log += f", TestLoss={test_loss:.4f}, {metric_name}={test_metric:.4f}"
        #epoch_log += f", TestLoss={test_loss:.4f}, {metric_name}={test_metric:.4f}, LR={current_lr:.6f}"

        if log_file:
            with open(log_file, "a") as f:
                f.write(epoch_log + "\n")
        print(epoch_log)

def evaluate(model, classifier, test_loader, device, is_regression=False):
    model.eval()
    classifier.eval() if classifier is not None else None
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            #print(f"DEBUG: x.shape={x.shape}, y.shape={y.shape}")
            all_pred, _ = model(x)
            if is_regression and classifier is None:
                y_pred = all_pred[-1].view(-1)
            else:
                y_pred = all_pred[-1]
            # deal with classifier for LSTM/RNN
            if classifier is not None:
                y_pred = classifier(y_pred)
                if is_regression:
                    y_pred = y_pred.view(-1)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            if is_regression:
                preds.append(y_pred.cpu().numpy())
                targets.append(y.cpu().numpy())
            else:
                total_acc += (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item() / len(y)
            n_batches += 1
    if is_regression:
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        print(f"Test Loss: {total_loss/n_batches:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        return total_loss/n_batches, rmse
    else:
        print(f"Test Loss: {total_loss/n_batches:.4f}")
        print(f"Test Accuracy: {total_acc/n_batches:.4f}")
        return total_loss/n_batches, total_acc/n_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", type=str, required=True, help="Dataset names (space separated, e.g. --dataset forda fordb)")
    parser.add_argument("--model_type", nargs="+", type=str, required=True, help="Model types (space separated, e.g. --model_type scn adadiag)")
    parser.add_argument("--input_size", type=int, help="Input size (overrides default)")
    parser.add_argument("--out_size", type=int, help="Output size (overrides default)")
    parser.add_argument("--block_sizes", nargs="+", type=int, default=32, help="Block sizes")
    parser.add_argument("--block_config", type=int, default=3, help="Block config")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function")
    parser.add_argument("--coupling_topology", type=int, default=20, help="Coupling topology")
    parser.add_argument("--block_init_fn", type=str, default="sparse", help="Block initializer function")
    parser.add_argument("--coupling_block_init_fn", type=str, default="uniform", help="Coupling block initializer function")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Test batch size")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs to use")
    parser.add_argument("--CPU", action="store_true", help="This forces to run on CPU, regardless of what is passed to --gpus")
    parser.add_argument("--blocksize32", action="store_true", help="This forces block sizes to be [32 for _ in range(16)], regardless of what is passed to --block_sizes")
    parser.add_argument("--blocksize8", action="store_true", help="This forces block sizes to be [8 for _ in range(64)], regardless of what is passed to --block_sizes")
    parser.add_argument("--eul_step", type=float, default=0.01, help="Euler step size")
    parser.add_argument("--gating", action="store_true", help="Use gating mechanism")
    parser.add_argument("--gate_bounds", nargs=2, type=float, default=[1e-5, 0.2], help="Min and max gate values")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[80, 160], help="Epochs at which to decay learning rate")
    parser.add_argument("--decay_scalar", type=float, default=0.1, help="Factor by which to decay learning rate")
    # LSTM/RNN specific args
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size (only for LSTM and RNN)")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM or RNN")
    parser.add_argument("--l2_regul", type=float, default=0., help="L2 regularization strength (only for LSTM and RNN)")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers (only for LSTM and RNN)")
    parser.add_argument("--nonlinearity", type=str, default="relu", help="Nonlinearity function (only for RNN)")
    args = parser.parse_args()


    # Example of usage:
    #python final_train_and_test.py --dataset forda --gpus 0 --lr 0.001 --trials 3 --model_type lstm --hidden_size 64 --n_layers 2 --bidirectional --l2_regul 0.


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Loop over all combinations of datasets and models
    for dataset_name in args.dataset:
        if dataset_name not in DATASET_LOADERS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        loader_fn, size_fn = DATASET_LOADERS[dataset_name]
        input_size, out_size = size_fn()
        if args.input_size is not None:
            input_size = args.input_size
        if args.out_size is not None:
            out_size = args.out_size

        is_regression = out_size == 1

        # Set CUDA device as specified by user, if available
        if args.CPU:
            device = torch.device("cpu")
        else:
            if args.gpus is not None and torch.cuda.is_available():
                device = torch.device(f"cuda:{args.gpus}")
            else:
                device = torch.device("cpu")
        print(f"Using device: {device}")

        # Load data
        if dataset_name == "har2":
            root = os.path.join(os.getcwd(), "data", "har")
        elif dataset_name in ["smnist", "psmnist"]:
            root = os.path.join(os.getcwd(), "data")
            permseed = 0 if dataset_name == "psmnist" else None
        else:
            root = None
        if dataset_name in ["smnist", "psmnist"]:
            full_train_loader, test_loader = loader_fn(
                    train_batch_size=args.train_batch_size,
                    test_batch_size=args.test_batch_size,
                    permute_seed=permseed,
                    root=root,
            )
        else:
            if dataset_name == "har2":
                train_loader, val_loader, test_loader = loader_fn(
                    train_batch_size=args.train_batch_size, 
                    test_batch_size=args.test_batch_size,
                    root=root,
                )
            else:
                train_loader, val_loader, test_loader = loader_fn(
                    train_batch_size=args.train_batch_size, 
                    test_batch_size=args.test_batch_size,
                    #root=root,
                )
            full_train_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset])
            full_train_loader = torch.utils.data.DataLoader(
                full_train_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                collate_fn=seq_collate_fn
            )

        for model_type in args.model_type:
            # Model-specific config
            block_init_fn_name = args.block_init_fn
            if model_type == "adadiag":
                if args.block_config == 3:
                    block_init_fn_name = "diagonal"
                else:
                    print(f"Warning: AdaDiag with block_config {args.block_config} has not been evaluated. Proceeding anyway.")
            elif model_type == "scn":
                if args.block_config == 1 or args.block_config == 2:
                    block_init_fn_name = "sparse"
                else:
                    print(f"Warning: SCN with block_config {args.block_config} is not standard. Proceeding anyway.")
            elif model_type == "lstm":
                pass
            elif model_type == "rnn":
                pass
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Initializer functions
            def get_initializer_fn(name):
                if name == "uniform":
                    def fn(shape, dtype):
                        return uniform(shape, dtype=dtype)
                    return fn
                elif name == "normal":
                    def fn(shape, dtype):
                        return normal(shape, dtype=dtype)
                    return fn
                elif name == "orthogonal":
                    return orthogonal
                elif name == "sparse":
                    def fn(shape, dtype):
                        return sparse(shape, dtype=dtype)
                    return fn
                elif name == "ones":
                    return ones
                elif name == "zeros":
                    return zeros
                elif name == "diagonal":
                    def fn(shape, dtype):
                        return diagonal(shape, dtype=dtype)
                    return fn
                elif name == "ring":
                    def fn(shape, dtype):
                        return ring(shape[0], dtype=dtype)
                    return fn
                elif name == "lower_feedforward":
                    return lower_feedforward
                elif name == "lower_triangular":
                    return lower_triangular
                else:
                    return orthogonal

            block_init_fn = get_initializer_fn(block_init_fn_name)
            coupling_block_init_fn = get_initializer_fn(args.coupling_block_init_fn)

            if args.blocksize32:
                args.block_sizes = [32 for _ in range(16)]
            if args.blocksize8:
                args.block_sizes = [8 for _ in range(64)]

            log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "final_search")
            os.makedirs(log_dir, exist_ok=True)
            bsize = str(args.train_batch_size)
            # change log file name to not overwrite previous results
            log_file = os.path.join(log_dir, f"results_{dataset_name}_{model_type}_batch{bsize}.txt")
            with open(log_file, "w") as f:
                f.write("==== Model Run Details ====\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Model type: {model_type}\n")
                f.write(f"Block sizes: {args.block_sizes}\n") if model_type in ["adadiag", "scn"] else None
                f.write(f"Block config: {args.block_config}\n") if model_type in ["adadiag", "scn"] else None
                f.write(f"Activation: {args.activation}\n")
                f.write(f"Coupling topology: {args.coupling_topology}\n") if model_type in ["adadiag", "scn"] else None
                f.write(f"Block init fn: {block_init_fn_name}\n")
                f.write(f"Coupling block init fn: {args.coupling_block_init_fn}\n") if model_type in ["adadiag", "scn"] else None
                f.write(f"Train batch size: {args.train_batch_size}\n")
                f.write(f"Test batch size: {args.test_batch_size}\n")
                f.write(f"Epochs: {args.epochs}\n")
                f.write(f"Decay epochs: {args.decay_epochs}\n")
                f.write(f"Decay scalar: {args.decay_scalar}\n")
                f.write(f"Gating: {args.gating}\n") if model_type in ["adadiag", "scn"] else None
                f.write(f"Gate bounds: {args.gate_bounds}\n") if model_type in ["adadiag", "scn"] else None
                f.write(f"Euler step: {args.eul_step}\n") if model_type in ["adadiag", "scn"] else None
                f.write(f"Hidden size: {args.hidden_size}\n") if model_type in ["lstm", "rnn"] else None
                f.write(f"Bidirectional: {args.bidirectional}\n") if model_type in ["lstm", "rnn"] else None
                f.write(f"Number of layers: {args.n_layers}\n") if model_type in ["lstm", "rnn"] else None
                f.write(f"L2 regularization: {args.l2_regul}\n") if model_type in ["lstm", "rnn"] else None
                f.write(f"Learning rate: {args.lr}\n")
                f.write(f"Trials: {args.trials}\n")
                f.write("==========================\n\n")

            test_scores = []
            best_trial_scores = []
            for trial in range(args.trials):
                if model_type in ["adadiag", "scn"]:
                    model = RNNAssembly.from_initializers(
                        input_size=input_size,
                        out_size=out_size,
                        block_sizes=args.block_sizes,
                        block_init_fn=block_init_fn,
                        coupling_block_init_fn=coupling_block_init_fn,
                        coupling_topology=args.coupling_topology,
                        activation=args.activation,
                        constrained_blocks="tanh",
                        coupling_rescaling = "unconstrained",
                        structure = "identity",
                        eul_step=args.eul_step,
                        gated_eul=args.gating,
                        min_gate=args.gate_bounds[0],
                        max_gate=args.gate_bounds[1],
                    )
                    classifier = None
                elif model_type == "lstm":
                    model = nn.LSTM(
                        input_size=input_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.n_layers,
                        bidirectional=args.bidirectional,
                        batch_first=False,
                        #nonlinearity=args.nonlinearity,
                    )
                    classifier = nn.Linear(
                        args.hidden_size * (2 if args.bidirectional else 1),
                        out_size
                    )
                elif model_type == "rnn":
                    model = nn.RNN(
                        input_size=input_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.n_layers,
                        bidirectional=args.bidirectional,
                        batch_first=False,
                        nonlinearity=args.nonlinearity,
                    )
                    classifier = nn.Linear(
                        args.hidden_size * (2 if args.bidirectional else 1),
                        out_size
                    )
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                model = model.to(device)
                classifier = classifier.to(device) if model_type in ["lstm", "rnn"] else None
                with open(log_file, "a") as f:
                    f.write(f"\n--- Trial {trial+1}/{args.trials} ---\n")
                epoch_scores = []
                train(
                    model,
                    full_train_loader,
                    test_loader,
                    device,
                    classifier=classifier if model_type in ["lstm", "rnn"] else None,
                    epochs=args.epochs,
                    is_regression=is_regression,
                    lr=args.lr,
                    wd=args.l2_regul if model_type in ["lstm", "rnn"] else 0.0,
                    decay_epochs=args.decay_epochs,
                    decay_scalar=args.decay_scalar,
                    log_file=log_file,
                    epoch_best_scores=epoch_scores
                )
                if is_regression:
                    best_score = np.min(epoch_scores)
                else:
                    best_score = np.max(epoch_scores)
                best_trial_scores.append(best_score)
                _, score = evaluate(model,
                                    classifier,
                                     test_loader, device, is_regression=is_regression)
                test_scores.append(score)
                trial_log = f"Trial {trial+1}/{args.trials}: FinalEpochScore={score:.4f}, BestEpochScore={best_score:.4f}"
                with open(log_file, "a") as f:
                    f.write(trial_log + "\n")
                print(trial_log)

            mean_score = np.mean(test_scores)
            std_score = np.std(test_scores)
            metric_name = "Test RMSE" if is_regression else "Test Accuracy"
            summary_log = f"{metric_name} over {args.trials} trials (final epoch): Mean={mean_score:.4f}, Std={std_score:.4f}"
            best_mean = np.mean(best_trial_scores)
            best_std = np.std(best_trial_scores)
            best_summary_log = f"{metric_name} over {args.trials} trials (best epoch): Mean={best_mean:.4f}, Std={best_std:.4f}"
            with open(log_file, "a") as f:
                f.write("\n==== FINAL SUMMARY ====\n")
                f.write(summary_log + "\n")
                f.write(best_summary_log + "\n")
            print(summary_log)
            print(best_summary_log)

if __name__ == "__main__":
    main()
