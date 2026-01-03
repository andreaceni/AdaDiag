#!/bin/bash


# Use the following to first make it executable and then run it
# chmod +x mediumbatches.sh
# ./mediumbatches.sh





# Medium batch sizes:
python final_train_and_test.py --dataset forda --gpus 1 --blocksize8 --block_config 3 --coupling_topology 20 --lr 0.001 --gating --trials 3 --model_type adadiag --train_batch_size 32 --test_batch_size 128
python final_train_and_test.py --dataset forda --gpus 1 --blocksize32 --eul_step 0.1 --block_config 2 --coupling_topology 500 --lr 0.001 --trials 3 --model_type scn --train_batch_size 32 --test_batch_size 128

python final_train_and_test.py --dataset fordb --gpus 1 --blocksize32 --block_config 3 --gating --coupling_topology 5 --lr 0.01 --trials 3 --model_type adadiag --train_batch_size 32 --test_batch_size 128
python final_train_and_test.py --dataset fordb --gpus 1 --blocksize32 --block_config 2 --coupling_topology 20 --eul_step 0.1 --lr 0.001 --trials 3 --model_type scn --train_batch_size 32 --test_batch_size 128

#python final_train_and_test.py --dataset ieeeppg --gpus 3 --blocksize8 --block_config 3 --gating --coupling_topology 5 --lr 0.01 --trials 3 --model_type adadiag --epochs 50 --train_batch_size 32 --test_batch_size 128
#python final_train_and_test.py --dataset ieeeppg --gpus 3 --blocksize32 --block_config 2 --eul_step 0.1 --coupling_topology 5 --lr 0.01 --trials 3 --model_type scn --epochs 50 --train_batch_size 32 --test_batch_size 128

#python final_train_and_test.py --dataset har2 --gpus 3 --blocksize8 --block_config 3 --gating --coupling_topology 20 --lr 0.01 --trials 3 --model_type adadiag --train_batch_size 32 --test_batch_size 128
#python final_train_and_test.py --dataset har2 --gpus 3 --blocksize32 --block_config 2 --eul_step 0.1 --coupling_topology 20 --lr 0.001 --trials 3 --model_type scn --train_batch_size 32 --test_batch_size 128

