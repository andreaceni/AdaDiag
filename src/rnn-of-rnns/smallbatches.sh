#!/bin/bash


# Use the following to first make it executable and then run it
# chmod +x smallbatches.sh
# ./smallbatches.sh





# Smaller batch sizes:
python final_train_and_test.py --dataset japanesevowels --gpus 3 --blocksize32 --block_config 3 --eul_step 0.1 --coupling_topology 20 --lr 0.01 --gating --trials 3 --model_type adadiag --train_batch_size 16 --test_batch_size 128
python final_train_and_test.py --dataset japanesevowels --gpus 3 --blocksize32 --block_config 1 --eul_step 0.1 --coupling_topology 20 --lr 0.01 --trials 3 --model_type scn --train_batch_size 16 --test_batch_size 128

python final_train_and_test.py --dataset adiac --gpus 3 --blocksize32 --block_config 3 --gating --coupling_topology 20 --lr 0.01 --trials 3 --model_type adadiag --train_batch_size 16 --test_batch_size 128
python final_train_and_test.py --dataset adiac --gpus 3 --blocksize32 --block_config 1 --coupling_topology 20 --eul_step 0.1 --lr 0.01 --trials 3 --model_type scn --train_batch_size 16 --test_batch_size 128

python final_train_and_test.py --dataset pems --gpus 3 --blocksize32 --block_config 3 --gating --coupling_topology 500 --lr 0.01 --trials 3 --model_type adadiag --train_batch_size 16 --test_batch_size 128
python final_train_and_test.py --dataset pems --gpus 3 --blocksize32 --block_config 2 --coupling_topology 20 --eul_step 0.01 --lr 0.01 --trials 3 --model_type scn --train_batch_size 16 --test_batch_size 128



