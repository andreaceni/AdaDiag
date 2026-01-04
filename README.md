# Sparse Assemblies of Recurrent Neural Networks with Stability Guarantees

Official implementation of the paper:

> **Ceni et al.**  
> *Sparse Assemblies of Recurrent Neural Networks with Stability Guarantees*  
> Neurocomputing, 2026.

## Repository Structure

- `src/rnn-of-rnns/torchdyno/models/rnn_assembly/`  
  Contains the core implementation of the proposed RNN assembly model.

- `src/rnn-of-rnns/search_space.py`  
  Defines the hyperparameter grid used for model selection.

- `src/rnn-of-rnns/main_search.py`  
  Runs the validation procedure via grid search.

- `src/rnn-of-rnns/final_train_and_test.py`  
  Performs final training and evaluation using the selected hyperparameters.

## Results

- Validation and test results are automatically saved in the `final_search/` directory.


## Example Usage (ADIAC dataset)

The following command runs the final training and evaluation on the **ADIAC** dataset:

```bash
python final_train_and_test.py \
  --dataset adiac \
  --gpus 0 \
  --blocksize32 \
  --block_config 3 \
  --gating \
  --coupling_topology 20 \
  --lr 0.01 \
  --trials 3 \
  --model_type adadiag \
  --train_batch_size 16
```

Argument Notes

--blocksize 32: number of recurrent units per module.

--coupling_topology 20: coupling parameter 
𝐶
C defined in the paper.
