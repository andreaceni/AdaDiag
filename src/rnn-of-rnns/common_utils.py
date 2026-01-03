import os
import numpy as np
import torch
from typing import Tuple



# MNIST loader (from utils.py)
def load_mnist(
    train_batch_size: int,
    test_batch_size: int,
    subset_train_size: int = None,
    subset_test_size: int = None,
    subset_seed: int = 42,
    **kwargs
):
    from torch.utils.data import DataLoader, Subset
    from torchdyno.data.utils.seq_loader import seq_collate_fn as mnist_seq_collate_fn
    from torchdyno.data.datasets.seq_mnist import SequentialMNIST

    train_data = SequentialMNIST(train=True, download=True, **kwargs)
    test_data = SequentialMNIST(train=False, download=True, **kwargs)

    model_selection = False
    if model_selection:
        # FORCE SUBSETTING FOR QUICKER MODEL SELECTION
        subset_train_size = 10000
        subset_test_size = 2000
  
    # Subsample if requested
    if subset_train_size is not None and subset_train_size < len(train_data):
        rng = np.random.RandomState(subset_seed)
        train_indices = rng.choice(len(train_data), subset_train_size, replace=False)
        train_data = Subset(train_data, train_indices)
    """
    if subset_test_size is not None and subset_test_size < len(test_data):
        rng = np.random.RandomState(subset_seed + 1)
        test_indices = rng.choice(len(test_data), subset_test_size, replace=False)
        test_data = Subset(test_data, test_indices)
    """
    if subset_test_size is not None and subset_test_size < len(train_data):
        rng = np.random.RandomState(subset_seed + 1)
        remaining_indices = list(set(range(len(SequentialMNIST(train=True))) ) - set(train_indices))
        subtest_indices = rng.choice(remaining_indices, subset_test_size, replace=False)
        test_data = Subset(SequentialMNIST(train=True, download=True, **kwargs), subtest_indices)


    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=mnist_seq_collate_fn(),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=mnist_seq_collate_fn(),
    )
    return train_loader, test_loader


def get_mnist_input_output_sizes():
    # MNIST is univariate, multi-class classification
    return 1, 10



# Shared collate function for HAR2, FordA, etc.
def seq_collate_fn(batch):
    """
    Collate function to produce [seq_len, batch_size, input_size] tensors.
    """
    Xs, ys = zip(*batch)
    X = torch.stack(Xs, dim=0)  # (batch, seq_len, feat_dim)
    y = torch.stack(ys)         # (batch,)

    # Convert labels to long only if they are integers (classification), otherwise float (regression)
    if y.dtype in [torch.int32, torch.int64]:
        y = y.long()  # classification
    else:
        y = y.float()  # regression
    X = X.permute(1, 0, 2)
    return X, y


def to_numpy3d(X):
    """Convert aeon dataset output to numpy 3D array (n_instances, n_channels, series_length)."""
    if isinstance(X, np.ndarray):
        return X.astype(np.float32)
    else:
        return np.asarray([x.values for x in X]).astype(np.float32)
    

def to_numpy3d_ieeeppg(X):
    """
    Convert IEEEPPG dataset output to numpy 3D array (n_samples, n_channels, seq_len).
    Pads sequences to the max length across all sequences.
    Handles variable-length and multivariate sequences.
    """
    X_list = []
    for x in X:
        arr = np.array(x, dtype=np.float32)
        # Ensure 2D: (n_channels, seq_len)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]  # 1 channel
        elif arr.ndim == 2:
            arr = arr  # already (n_channels, seq_len)
        else:
            raise ValueError(f"Unexpected shape for sequence: {arr.shape}")
        X_list.append(arr)

    # Find max sequence length
    max_len = max(arr.shape[1] for arr in X_list)
    n_channels = X_list[0].shape[0]
    n_samples = len(X_list)

    # Pad sequences with zeros
    X_padded = np.zeros((n_samples, n_channels, max_len), dtype=np.float32)
    for i, arr in enumerate(X_list):
        seq_len = arr.shape[1]
        X_padded[i, :, :seq_len] = arr

    return X_padded



# HAR2 utilities
def load_har2(root: str) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """
    Load HAR-2 dataset for Human Activity Recognition with binary classification.
    Dataset preprocessing code adapted from
    https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/LSTM.ipynb
    LABELS = [
        "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"
    ]
    Binary classes:
    - Class 0: {Walking Upstairs, Sitting, Laying}  
    - Class 1: {Walking, Walking Downstairs, Standing}
    Args:
        root (str): Root directory containing the HAR dataset files
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]
    CLASS_MAP = {1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0}
    TRAIN = "train"
    TEST = "test"

    def load_X(X_signals_paths):
        X_signals = []
        for signal_type_path in X_signals_paths:
            with open(signal_type_path, 'r') as file:
                X_signals.append(
                    [np.array(serie, dtype=np.float32) for serie in [
                        row.replace('  ', ' ').strip().split(' ') for row in file
                    ]]
                )
        return np.transpose(np.array(X_signals), (1, 2, 0))

    def load_y(y_path):
        with open(y_path, 'r') as file:
            y_ = np.array(
                [CLASS_MAP[int(row)] for row in file],
                dtype=np.int32
            )
        return y_

    X_train_signals_paths = [
        os.path.join(root, TRAIN, "Inertial Signals", signal + "train.txt") 
        for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        os.path.join(root, TEST, "Inertial Signals", signal + "test.txt") 
        for signal in INPUT_SIGNAL_TYPES
    ]

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)
    y_train = load_y(os.path.join(root, TRAIN, "y_train.txt"))
    y_test = load_y(os.path.join(root, TEST, "y_test.txt"))

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(), 
        torch.from_numpy(y_train).long()
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).long()
    )

    val_length = int(len(train_dataset) * 0.3)
    train_length = len(train_dataset) - val_length

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_length, val_length]
    )

    return train_dataset, val_dataset, test_dataset

def get_har2_input_output_sizes() -> Tuple[int, int]:
    """
    Get input and output sizes for HAR-2 dataset.
    Returns:
        Tuple of (input_size, output_size)
    """
    input_size = 9  # 9 sensor signals
    output_size = 2  # Binary classification
    return input_size, output_size

def load_har2_dataloaders(root: str, train_batch_size: int = 64, test_batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load HAR-2 dataset and return data loaders.
    Args:
        root (str): Root directory containing the HAR dataset files
        train_batch_size (int): Batch size for training
        test_batch_size (int): Batch size for validation and testing
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = load_har2(root)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=True, 
        drop_last=False,
        collate_fn=seq_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=seq_collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=seq_collate_fn,
    )
    return train_loader, val_loader, test_loader



# FordA utilities
def load_forda_dataloaders(train_batch_size: int = 64, test_batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load FordA dataset from aeon and return train/val/test DataLoaders.
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from aeon.datasets import load_classification
    X_train, y_train = load_classification("FordA", split="train", load_equal_length=False, load_no_missing=False)
    X_test, y_test = load_classification("FordA", split="test", load_equal_length=False, load_no_missing=False)

    X_train = to_numpy3d(X_train)
    X_test = to_numpy3d(X_test)
    y_train = to_numpy3d(y_train)
    y_test = to_numpy3d(y_test)

    y_train = (y_train + 1) // 2
    y_test = (y_test + 1) // 2

    val_size = int(0.3 * len(X_train))
    train_size = len(X_train) - val_size
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=seq_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)

    return train_loader, val_loader, test_loader

def get_forda_input_output_sizes():
    # FordA is univariate, binary classification
    return 1, 2


# FordB utilities
def load_fordb_dataloaders(train_batch_size: int = 64, test_batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load FordB dataset from aeon and return train/val/test DataLoaders.
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from aeon.datasets import load_classification
    X_train, y_train = load_classification("FordB", split="train", load_equal_length=True, load_no_missing=False)
    X_test, y_test = load_classification("FordB", split="test", load_equal_length=True, load_no_missing=False)

    X_train = to_numpy3d(X_train)
    X_test = to_numpy3d(X_test)
    y_train = to_numpy3d(y_train)
    y_test = to_numpy3d(y_test)

    y_train = (y_train + 1) // 2
    y_test = (y_test + 1) // 2

    val_size = int(0.3 * len(X_train))
    train_size = len(X_train) - val_size
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=seq_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)

    return train_loader, val_loader, test_loader

def get_fordb_input_output_sizes():
    # FordB is univariate, binary classification
    return 1, 2


# Adiac utilities
def load_adiac_dataloaders(train_batch_size: int = 64, test_batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ 
    Load Adiac dataset from aeon and return train/val/test DataLoaders.
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from aeon.datasets import load_classification
    X_train, y_train = load_classification("Adiac", split="train", load_equal_length=True, load_no_missing=False)
    X_test, y_test = load_classification("Adiac", split="test", load_equal_length=True, load_no_missing=False)

    X_train = to_numpy3d(X_train)
    X_test = to_numpy3d(X_test)
    y_train = to_numpy3d(y_train)
    y_test = to_numpy3d(y_test)

    # shift them into a [0, outsize] range (since PyTorch expects class indices starting at 0)
    y_train = y_train - 1
    y_test = y_test - 1

    # hold out 10% of training set for validation
    val_size = int(0.1 * len(X_train))
    train_size = len(X_train) - val_size
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    # check shapes
    #print("Adiac shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # Adiac shapes: (351, 1, 176) (351,) (391, 1, 176) (391,) 

    # adiac comes in shape (num_samples, 1, seq_len)
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1) # (num_samples, seq_len, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)    

    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=seq_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)

    return train_loader, val_loader, test_loader

def get_adiac_input_output_sizes():
    # Adiac is univariate, binary classification
    return 1, 37




# IEEEPPG utilities
def load_IEEEPPG_dataloaders(train_batch_size: int = 64, test_batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load IEEEPPG dataset from aeon and return train/val/test DataLoaders.
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from aeon.datasets import load_regression

    X, y = load_regression(
        "IEEEPPG", 
        load_equal_length=False, 
        load_no_missing=False, 
        return_metadata=False
    )

    # aeon returns a numpy array of shape (N_samples, inp_size, seq_len) 
    X = to_numpy3d_ieeeppg(X)
    y = np.array(y).astype(np.float32)
    print("IEEEPPG shapes:", X.shape, y.shape)  # 
    X = torch.tensor(X, dtype=torch.float32)  
    y = torch.from_numpy(y).float()  

    # Split into train/val/test sets (80/10/10)
    n = X.shape[0]
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size
    train_idx = torch.arange(0, train_size)
    val_idx = torch.arange(train_size, train_size + val_size)
    test_idx = torch.arange(train_size + val_size, n)

    # need to convert to (N_samples, seq_len, inp_size)  for torch dataloaders
    X_train = torch.tensor(X[train_idx, :], dtype=torch.float32).permute(0, 2, 1)
    X_val = torch.tensor(X[val_idx, :], dtype=torch.float32).permute(0, 2, 1)
    X_test = torch.tensor(X[test_idx, :], dtype=torch.float32).permute(0, 2, 1)

    train_dataset = torch.utils.data.TensorDataset(X_train, y[train_idx])
    val_dataset = torch.utils.data.TensorDataset(X_val, y[val_idx])
    test_dataset = torch.utils.data.TensorDataset(X_test, y[test_idx])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=seq_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=seq_collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=seq_collate_fn,
    )

    return train_loader, val_loader, test_loader


def get_IEEEPPG_input_output_sizes():
    # IEEEPPG is multivariate, regression
    return 5, 1






# NewsTitleSentiment utilities
def load_NewsTitleSentiment_dataloaders(train_batch_size: int = 64, test_batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load NewsTitleSentiment dataset from aeon and return train/val/test DataLoaders.
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from aeon.datasets import load_regression

    try:
        X, y = load_regression(
            "NewsTitleSentiment", 
            load_equal_length=False, 
            load_no_missing=False, 
            return_metadata=False
        )

        X_list = list(X)  # ensure each sample is an element
        # aeon returns a numpy array of shape (N_samples, inp_size, seq_len) 
        X = to_numpy3d_ieeeppg(X)
        y = np.array(y).astype(np.float32)
        print("NewsTitleSentiment shapes:", X.shape, y.shape)  #
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.from_numpy(y).float()

        # Split into train/val/test sets (80/10/10)
        n = X.shape[0]
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        test_size = n - train_size - val_size
        train_idx = torch.arange(0, train_size)
        val_idx = torch.arange(train_size, train_size + val_size)
        test_idx = torch.arange(train_size + val_size, n)

        # need to convert to (N_samples, seq_len, inp_size)  for torch dataloaders
        X_train = torch.tensor(X[train_idx, :], dtype=torch.float32).permute(0, 2, 1)
        X_val = torch.tensor(X[val_idx, :], dtype=torch.float32).permute(0, 2, 1)
        X_test = torch.tensor(X[test_idx, :], dtype=torch.float32).permute(0, 2, 1)

        train_dataset = torch.utils.data.TensorDataset(X_train, y[train_idx])
        val_dataset = torch.utils.data.TensorDataset(X_val, y[val_idx])
        test_dataset = torch.utils.data.TensorDataset(X_test, y[test_idx])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=seq_collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=seq_collate_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=seq_collate_fn,
        )

        return train_loader, val_loader, test_loader

    except OSError as e:
        print(f"Skipping NewsTitleSentiment due to load error: {e}")
        return None, None, None


def get_NewsTitleSentiment_input_output_sizes():
    # NewsTitleSentiment is multivariate, regression
    return 3, 1




# JapaneseVowels utilities
def load_JapaneseVowels_dataloaders(train_batch_size: int = 64, test_batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load JapaneseVowels dataset from aeon and return train/val/test DataLoaders.
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from aeon.datasets import load_classification
    X_train, y_train = load_classification("JapaneseVowels", split="train", load_equal_length=True, load_no_missing=False)
    X_test, y_test = load_classification("JapaneseVowels", split="test", load_equal_length=True, load_no_missing=False)

    X_train = to_numpy3d(X_train)
    X_test = to_numpy3d(X_test)
    y_train = to_numpy3d(y_train)
    y_test = to_numpy3d(y_test)

    # shift them into a [0, outsize] range (since PyTorch expects class indices starting at 0)
    y_train = y_train - 1
    y_test = y_test - 1

    val_size = int(0.3 * len(X_train))
    train_size = len(X_train) - val_size
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=seq_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)

    return train_loader, val_loader, test_loader

def get_JapaneseVowels_input_output_sizes():
    # JapaneseVowels is multivariate, multi-class classification
    return 12, 9



# PEMS-SF utilities
def load_PEMS_SF_dataloaders(train_batch_size: int = 64, test_batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load PEMS-SF dataset from aeon and return train/val/test DataLoaders.
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from aeon.datasets import load_classification
    import numpy as np
    import torch

    # Load train/test split
    X_train, y_train = load_classification("PEMS-SF", split="train", load_equal_length=True, load_no_missing=False)
    X_test, y_test = load_classification("PEMS-SF", split="test", load_equal_length=True, load_no_missing=False)

    # Convert to numpy 3D arrays [n_samples, n_channels, series_length]
    X_train = to_numpy3d(X_train)
    X_test = to_numpy3d(X_test)

    # Labels can sometimes be strings like "3.0" → cast to float → int
    y_train = np.asarray(y_train, dtype=float).astype(int)
    y_test = np.asarray(y_test, dtype=float).astype(int)

    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    # Shift labels to start from 0
    y_train -= y_train.min()
    y_test -= y_test.min()

    # Train/val split
    val_size = int(0.3 * len(X_train))
    train_size = len(X_train) - val_size
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    # Convert to tensors [batch, channels, length]
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()

    # Wrap in TensorDataset
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # Build DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=seq_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)

    return train_loader, val_loader, test_loader


def get_PEMS_SF_input_output_sizes():
    # PEMS-SF is multivariate with 963 channels and 7 output classes
    return 963, 7





###############################################################################################
# Utilities for Heartbeat, UWaveGestureLibrary
###############################################################################################

# Heartbeat
def load_Heartbeat_dataloaders(train_batch_size=64, test_batch_size=64):
    return load_generic_dataloaders("Heartbeat", train_batch_size, test_batch_size)

def get_Heartbeat_input_output_sizes():
    return 61, 2   



# UWaveGestureLibrary
def load_UWaveGestureLibrary_dataloaders(train_batch_size=64, test_batch_size=64):
    return load_generic_dataloaders("UWaveGestureLibrary", train_batch_size, test_batch_size)

def get_UWaveGestureLibrary_input_output_sizes():
    return 3, 8   # 3 accelerometer channels, 8 gesture classes

###################################################
# Shared utility for loading datasets in my format

def clean_labels(y):
    """Convert labels to contiguous int indices [0..C-1]."""
    # If labels are strings like '1.0', '2.0', convert to float -> int
    try:
        y = np.array([int(float(lbl)) for lbl in y], dtype=np.int64)
    except Exception:
        # fallback: categorical string labels -> map to int
        classes = sorted(list(set(y)))
        mapping = {c: i for i, c in enumerate(classes)}
        y = np.array([mapping[lbl] for lbl in y], dtype=np.int64)
    return y


def load_generic_dataloaders(dataset_name: str, 
                             train_batch_size: int = 64, 
                             test_batch_size: int = 64,
                             label_mapping: dict = None
                             ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Generic loader for aeon classification datasets, returns train/val/test DataLoaders.
    Args:
        dataset_name: str, dataset name from aeon (e.g. "Heartbeat", "FaceDetection")
        train_batch_size: batch size for training
        test_batch_size: batch size for val/test
        label_mapping: optional dict to map string labels -> int
    """
    from aeon.datasets import load_classification
    X_train, y_train = load_classification(dataset_name, split="train", load_equal_length=True, load_no_missing=False)
    X_test, y_test = load_classification(dataset_name, split="test", load_equal_length=True, load_no_missing=False)

    # map labels if needed
    if label_mapping is not None:
        y_train = np.array([label_mapping[y] for y in y_train], dtype=np.int64)
        y_test  = np.array([label_mapping[y] for y in y_test], dtype=np.int64)
    else:
        y_train = clean_labels(y_train)
        y_test  = clean_labels(y_test)


    # convert to numpy arrays (aeon returns nested structures sometimes)
    X_train = np.array(X_train, dtype=np.float32)
    X_test  = np.array(X_test, dtype=np.float32)

    # check shapes
    print('#######################################################################\n')
    print(f"TRAIN {dataset_name} shapes:", X_train.shape, y_train.shape)  #
    print(f"TEST {dataset_name} shapes:", X_test.shape, y_test.shape)  #
    print('#######################################################################\n')


    # make validation split (30%)
    val_size = int(0.3 * len(X_train))
    train_size = len(X_train) - val_size
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    # torch tensors, shapes (N_samples, seq_len, inp_size)
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val   = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    X_test  = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.from_numpy(y_train).long()
    y_val   = torch.from_numpy(y_val).long()
    y_test  = torch.from_numpy(y_test).long()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset   = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset  = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=seq_collate_fn)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=seq_collate_fn)

    return train_loader, val_loader, test_loader



def sparsity_perc(M,N,C,S,mode='scn'):
    """
    Compute sparsity percentage for a given layer configuration.
    Args:
        M: int, number of modules
        N: int, module size
        C: int, number of coupling blocks
        S: float, sparsity level (0 < S < 1), only for 'scn' mode
        mode: str, 'scn' or 'adadiag'
    Returns:
        sparsity percentage (float)
    """
    total = (M*N)**2
    if mode == 'scn':
        diag = M * N**2 * S  # diagonal blocks
        offdiag = 2 * C * N**2     # off-diagonal blocks (both upper and lower)
        sparsity = (diag + offdiag) / total
    elif mode == 'adadiag':
        diag = M * (1/N)   # weights + biases
        offdiag = 2 * C * N**2     # off-diagonal blocks (both upper and lower)
        sparsity = (diag + offdiag) / total
    else:
        raise ValueError("Invalid mode. Choose 'scn' or 'adadiag'.")
    return sparsity
