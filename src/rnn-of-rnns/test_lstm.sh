#!/bin/bash


# Use the following to first make it executable and then run it
# chmod +x test_lstm.sh
# ./test_lstm.sh


# Running:
#python final_train_and_test.py --trials 3 --model_type lstm --dataset japanesevowels --gpus 1 --lr 0.001 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0 --train_batch_size 16
#python final_train_and_test.py --trials 3 --model_type lstm --dataset adiac --gpus 1 --lr 0.001 --hidden_size 128 --n_layers 2 --bidirectional --l2_regul 0 --train_batch_size 16
#python final_train_and_test.py --trials 3 --model_type lstm --dataset pems --gpus 1 --lr 0.001 --hidden_size 128 --n_layers 2 --bidirectional --l2_regul 0 --train_batch_size 16
#python final_train_and_test.py --trials 3 --model_type lstm --dataset forda --gpus 1 --lr 0.01 --hidden_size 64 --n_layers 3 --bidirectional --l2_regul 0
#python final_train_and_test.py --trials 3 --model_type lstm --dataset fordb --gpus 1 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0
python final_train_and_test.py --trials 3 --model_type lstm --dataset ieeeppg --gpus 1 --lr 0.001 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0 --epochs 50 --decay_epochs 20 40
python final_train_and_test.py --trials 3 --model_type lstm --dataset har2 --gpus 1 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0
python final_train_and_test.py --trials 3 --model_type lstm --dataset newstitlesentiment --gpus 1 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0.0001 --epochs 50 --decay_epochs 20 40
python final_train_and_test.py --trials 3 --model_type lstm --dataset smnist --gpus 1 --lr 0.01 --hidden_size 128 --n_layers 2 --l2_regul 0
python final_train_and_test.py --trials 3 --model_type lstm --dataset psmnist --gpus 1 --lr 0.001 --hidden_size 64 --n_layers 3 --bidirectional --l2_regul 0

