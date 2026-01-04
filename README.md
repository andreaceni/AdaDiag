# AdaDiag
Official repository of Ceni et al. "Sparse Assemblies of Recurrent Neural Networks with Stability Guarantees" Neurocomputing (2026).

The actual implementation of the model can be found in the folder ```src/rnn-of-rnns/torchdyno/models/rnn_assembly```.

Validation has been done via the ```src/rnn-of-rnns/main_search.py``` script using the grid search specified in the ```src/rnn-of-rnns/search_space.py``` file.

Final results have been obtained via the ```src/rnn-of-rnns/final_train_and_test.py``` script.

Results are automatically stored in the ```final_search/``` folder.

## Example of usage on Adiac

```python final_train_and_test.py --dataset adiac --gpus 0 --blocksize32 --block_config 3 --gating --coupling_topology 20 --lr 0.01 --trials 3 --model_type adadiag --train_batch_size 16```

--blocksize32 means that modules have 32 recurrent units

--coupling_topology corresponds to the C parameter of the paper
