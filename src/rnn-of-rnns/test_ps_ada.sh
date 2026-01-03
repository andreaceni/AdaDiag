#!/bin/bash


# Use the following to first make it executable and then run it
# chmod +x test_experiment.sh
# ./test_experiment.sh





#python final_train_and_test.py --dataset psmnist --gpus 0 --blocksize32 --block_config 3 --gating --coupling_topology 20 --lr 0.01 --trials 3 --model_type adadiag --train_batch_size 512 --test_batch_size 512

#python final_train_and_test.py --dataset psmnist --gpus 1 --blocksize32 --block_config 2 --coupling_topology 20 --eul_step 0.1 --lr 0.01 --trials 3 --model_type scn --train_batch_size 512 --test_batch_size 512

#python final_train_and_test.py --dataset smnist --gpus 2 --blocksize32 --block_config 3 --gating --coupling_topology 20 --lr 0.01 --trials 3 --model_type adadiag --train_batch_size 512 --test_batch_size 512

#python final_train_and_test.py --dataset smnist --gpus 3 --blocksize32 --block_config 2 --coupling_topology 500 --eul_step 0.1 --lr 0.001 --trials 3 --model_type scn --train_batch_size 512 --test_batch_size 512




# test the best version with lr=1e-3 and replacing the batchsize of 512 with 128
python final_train_and_test.py --dataset psmnist --gpus 0 --blocksize32 --block_config 3 --gating --coupling_topology 20 --lr 0.001 --trials 3 --model_type adadiag --train_batch_size 128 --test_batch_size 128
