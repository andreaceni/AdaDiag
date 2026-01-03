#!/bin/bash


# Use the following to first make it executable and then run it
# chmod +x test_experiment.sh
# ./test_experiment.sh

# Completed
#
#python final_train_and_test.py --dataset forda --gpus 2 --blocksize8 --block_config 3 --coupling_topology 20 --lr 0.001 --gating --trials 3 --model_type adadiag
#python final_train_and_test.py --dataset forda --gpus 2 --blocksize32 --eul_step 0.1 --block_config 2 --coupling_topology 500 --lr 0.001 --trials 3 --model_type scn
#
#python final_train_and_test.py --dataset har2 --gpus 2 --blocksize8 --block_config 3 --gating --coupling_topology 20 --lr 0.01 --trials 3 --model_type adadiag
#python final_train_and_test.py --dataset har2 --gpus 2 --blocksize32 --block_config 2 --eul_step 0.1 --coupling_topology 20 --lr 0.001 --trials 3 --model_type scn 
#
#python final_train_and_test.py --dataset ieeeppg --gpus 2 --blocksize8 --block_config 3 --gating --coupling_topology 5 --lr 0.01 --trials 3 --model_type adadiag --epochs 50
#python final_train_and_test.py --dataset ieeeppg --gpus 2 --blocksize32 --block_config 2 --eul_step 0.1 --coupling_topology 5 --lr 0.01 --trials 3 --model_type scn --epochs 50
#
#python final_train_and_test.py --dataset fordb --gpus 2 --blocksize32 --block_config 3 --gating --coupling_topology 5 --lr 0.01 --trials 3 --model_type adadiag
#python final_train_and_test.py --dataset fordb --gpus 2 --blocksize32 --block_config 2 --coupling_topology 20 --eul_step 0.1 --lr 0.001 --trials 3 --model_type scn
#
#python final_train_and_test.py --dataset newstitlesentiment --epochs 50 --decay_epochs 20 40 --train_batch_size 1024 --test_batch_size 1024 --gpus 2 --blocksize8 --block_config 3 --gating --coupling_topology 5 --lr 0.001 --trials 3 --model_type adadiag
#python final_train_and_test.py --dataset newstitlesentiment --epochs 50 --decay_epochs 20 40 --train_batch_size 1024 --test_batch_size 1024 --gpus 2 --blocksize8 --block_config 2 --coupling_topology 5 --eul_step 0.01 --lr 0.001 --trials 3 --model_type scn
#
#
# Smaller batch sizes::
#python final_train_and_test.py --dataset japanesevowels --gpus 2 --blocksize32 --block_config 3 --eul_step 0.1 --coupling_topology 20 --lr 0.01 --gating --trials 3 --model_type adadiag --train_batch_size 16 --test_batch_size 128
#python final_train_and_test.py --dataset japanesevowels --gpus 2 --blocksize32 --block_config 1 --eul_step 0.1 --coupling_topology 20 --lr 0.01 --trials 3 --model_type scn --train_batch_size 16 --test_batch_size 128
#
#python final_train_and_test.py --dataset adiac --gpus 2 --blocksize32 --block_config 3 --gating --coupling_topology 20 --lr 0.01 --trials 3 --model_type adadiag --train_batch_size 16 --test_batch_size 128
#python final_train_and_test.py --dataset adiac --gpus 2 --blocksize32 --block_config 1 --coupling_topology 20 --eul_step 0.1 --lr 0.01 --trials 3 --model_type scn --train_batch_size 16 --test_batch_size 128
#
#python final_train_and_test.py --dataset pems --gpus 2 --blocksize32 --block_config 3 --gating --coupling_topology 500 --lr 0.01 --trials 3 --model_type adadiag --train_batch_size 16 --test_batch_size 128
#python final_train_and_test.py --dataset pems --gpus 2 --blocksize32 --block_config 2 --coupling_topology 20 --eul_step 0.01 --lr 0.01 --trials 3 --model_type scn --train_batch_size 16 --test_batch_size 128
#


# Running in parallel as different scripts
#python final_train_and_test.py --dataset psmnist --gpus 0 --blocksize32 --block_config 3 --gating --coupling_topology 20 --lr 0.01 --trials 3 --model_type adadiag
#python final_train_and_test.py --dataset psmnist --gpus 1 --blocksize32 --block_config 2 --coupling_topology 20 --eul_step 0.1 --lr 0.01 --trials 3 --model_type scn
#python final_train_and_test.py --dataset smnist --gpus 2 --blocksize32 --block_config 3 --gating --coupling_topology 20 --lr 0.01 --trials 3 --model_type adadiag
#python final_train_and_test.py --dataset smnist --gpus 3 --blocksize32 --block_config 2 --coupling_topology 500 --eul_step 0.1 --lr 0.001 --trials 3 --model_type scn



# Running (LSTM and RNN):
python final_train_and_test.py --trials 3 --model_type lstm --dataset japanesevowels --gpus 1 --lr 0.001 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0 --train_batch_size 16
python final_train_and_test.py --trials 3 --model_type lstm --dataset adiac --gpus 1 --lr 0.001 --hidden_size 128 --n_layers 2 --bidirectional --l2_regul 0 --train_batch_size 16
python final_train_and_test.py --trials 3 --model_type lstm --dataset pems --gpus 1 --lr 0.001 --hidden_size 128 --n_layers 2 --bidirectional --l2_regul 0 --train_batch_size 16
python final_train_and_test.py --trials 3 --model_type lstm --dataset forda --gpus 1 --lr 0.01 --hidden_size 64 --n_layers 3 --bidirectional --l2_regul 0
python final_train_and_test.py --trials 3 --model_type lstm --dataset fordb --gpus 1 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0
python final_train_and_test.py --trials 3 --model_type lstm --dataset ieeeppg --gpus 1 --lr 0.001 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0
python final_train_and_test.py --trials 3 --model_type lstm --dataset har2 --gpus 1 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0
python final_train_and_test.py --trials 3 --model_type lstm --dataset newstitlesentiment --gpus 1 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0.0001
python final_train_and_test.py --trials 3 --model_type lstm --dataset smnist --gpus 1 --lr 0.01 --hidden_size 128 --n_layers 2 --l2_regul 0
python final_train_and_test.py --trials 3 --model_type lstm --dataset psmnist --gpus 1 --lr 0.001 --hidden_size 64 --n_layers 3 --bidirectional --l2_regul 0

python final_train_and_test.py --trials 3 --model_type rnn --dataset japanesevowels --gpus 3 --lr 0.001 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0.0001 --train_batch_size 16
python final_train_and_test.py --trials 3 --model_type rnn --dataset adiac --gpus 3 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0.0001 --train_batch_size 16
python final_train_and_test.py --trials 3 --model_type rnn --dataset pems --gpus 3 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0 --train_batch_size 16
python final_train_and_test.py --trials 3 --model_type rnn --dataset forda --gpus 3 --lr 0.01 --hidden_size 128 --n_layers 2 --bidirectional --l2_regul 0
python final_train_and_test.py --trials 3 --model_type rnn --dataset fordb --gpus 3 --lr 0.01 --hidden_size 64 --n_layers 2 --l2_regul 0.0001
python final_train_and_test.py --trials 3 --model_type rnn --dataset ieeeppg --gpus 3 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0
python final_train_and_test.py --trials 3 --model_type rnn --dataset har2 --gpus 3 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0.0001
python final_train_and_test.py --trials 3 --model_type rnn --dataset newstitlesentiment --gpus 3 --lr 0.001 --hidden_size 64 --n_layers 3 --bidirectional --l2_regul 0.0001
python final_train_and_test.py --trials 3 --model_type rnn --dataset smnist --gpus 3 --lr 0.001 --hidden_size 64 --n_layers 3 --bidirectional --l2_regul 0
python final_train_and_test.py --trials 3 --model_type rnn --dataset psmnist --gpus 3 --lr 0.001 --hidden_size 128 --n_layers 2 --bidirectional --l2_regul 0



