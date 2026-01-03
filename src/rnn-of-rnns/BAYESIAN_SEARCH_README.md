# Bayesian Search Implementation for Enhanced RNNAssembly

This document describes the new Bayesian search implementation for RNNAssembly experiments.

## New Files Created

1. **enhanced_rnn_assembly.py** - Enhanced RNNAssembly model with LayerNorm and attention pooling
2. **bayesian_trainable.py** - Bayesian trainable class with AdamW optimizer  
3. **bayesian_search.py** - Main Bayesian search script
4. **search_space.py** - Updated with bayesian_rnnassembly_search_space function
5. **test_bayesian_implementation.py** - Test script for verification

## Key Features

### Enhanced RNNAssembly Model
- LayerNorm stabilization in recurrent cells
- Attention-based pooling using dot-product attention
- Learnable query vector shared across time steps
- Formula: alpha_t = softmax(q^T h_t)

### Bayesian Trainable
- AdamW optimizer with weight decay optimization (0 to 1e-3)
- Advanced learning rate scheduling (cosine, onecycle, plateau)
- Enhanced parameter counting including new components

### Search Space Parameters
- Weight Decay: 1e-6 to 1e-3 (log-uniform)
- Learning Rate: 1e-4 to 1e-1 (log-uniform) 
- Gate Bounds: a(1e-6 to 1e-2), b(1e-2 to 1.0) (log-uniform)
- Coupling Topology: 2 to 1000 (quantized log-uniform)
- Block Sizes: [A for _ in range(B)] where A:2-512, B:2-128
- LR Scheduler: choice of cosine, onecycle, plateau
- LayerNorm: True/False
- Attention Pooling: True/False

## Usage

### Basic Usage
```bash
python bayesian_search.py --datasets fordb --n_trials 50
```

### Multiple Datasets
```bash
python bayesian_search.py --datasets fordb forda har2 --n_trials 100
```

### GPU Configuration
```bash
python bayesian_search.py --datasets fordb --gpus 0,1 --num_gpus 1.0 --num_cpus 8
```

### Advanced Options
```bash
python bayesian_search.py --datasets fordb \
    --results_dir /path/to/results \
    --n_trials 200 \
    --checkpoint \
    --checkpoint_metric test_acc \
    --max_epochs 300
```

## Implementation Details

### Architecture
- Input -> RNN Blocks (with LayerNorm) -> Attention Pooling -> Output
- LayerNorm applied after each recurrent update
- Attention mechanism aggregates all hidden states

### Parameter Count Changes
- LayerNorm: 2 * hidden_size parameters
- Attention query: hidden_size parameters  
- Total addition: 3 * hidden_size parameters

### Bayesian Optimization
- Uses OptunaSearch for efficient hyperparameter optimization
- ASHAScheduler for early stopping
- Optimizes test_acc for classification, test_loss for regression

## Supported Datasets
- har2, forda, fordb, adiac, japanesevowels
- ieeeppg, newstitlesentiment (regression)
- smnist, psmnist, pems

## Testing
```bash
python test_bayesian_implementation.py
```

## Integration
- Non-destructive: separate from existing codebase
- Uses same data loaders and utilities
- Compatible with existing main_search.py
- All new components clearly marked for Bayesian search

## Performance Expectations
- Better training stability with LayerNorm
- Improved sequence modeling with attention pooling
- Better optimization with AdamW and advanced scheduling
- Minimal computational overhead

## Troubleshooting
- Ensure dependencies: ray[tune], torch, optuna
- Start with small trials for debugging: --n_trials 5 --max_epochs 10
- Check GPU memory if using large models