#!/bin/bash


# Use the following to first make it executable and then run it
# chmod +x test_rnn.sh
# ./test_rnn.sh


# Running:
#python final_train_and_test.py --trials 3 --model_type rnn --dataset japanesevowels --gpus 3 --lr 0.001 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0.0001 --train_batch_size 16
#python final_train_and_test.py --trials 3 --model_type rnn --dataset adiac --gpus 3 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0.0001 --train_batch_size 16
#python final_train_and_test.py --trials 3 --model_type rnn --dataset pems --gpus 3 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0 --train_batch_size 16
#python final_train_and_test.py --trials 3 --model_type rnn --dataset forda --gpus 3 --lr 0.01 --hidden_size 128 --n_layers 2 --bidirectional --l2_regul 0
#python final_train_and_test.py --trials 3 --model_type rnn --dataset fordb --gpus 3 --lr 0.01 --hidden_size 64 --n_layers 2 --l2_regul 0.0001
python final_train_and_test.py --trials 3 --model_type rnn --dataset ieeeppg --gpus 3 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0 --epochs 50 --decay_epochs 20 40
python final_train_and_test.py --trials 3 --model_type rnn --dataset har2 --gpus 3 --lr 0.01 --hidden_size 128 --n_layers 3 --bidirectional --l2_regul 0.0001
python final_train_and_test.py --trials 3 --model_type rnn --dataset newstitlesentiment --gpus 3 --lr 0.001 --hidden_size 64 --n_layers 3 --bidirectional --l2_regul 0.0001 --epochs 50 --decay_epochs 20 40
python final_train_and_test.py --trials 3 --model_type rnn --dataset smnist --gpus 3 --lr 0.001 --hidden_size 64 --n_layers 3 --bidirectional --l2_regul 0
python final_train_and_test.py --trials 3 --model_type rnn --dataset psmnist --gpus 3 --lr 0.001 --hidden_size 128 --n_layers 2 --bidirectional --l2_regul 0

