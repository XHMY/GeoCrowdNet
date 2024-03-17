for seed in 0 1 2 3 4
do
    python main.py --accelerator cpu --experiment_name music_F \
    --regularization_type F --K 10 --lambda_reg 0.01 --batch_size 64 --n_epoch 1000 --seed $seed &

    python main.py --accelerator cpu --experiment_name music_no \
    --K 10 --regularization_type no --n_epoch 1000 --batch_size 64 --seed $seed &

    python main.py --accelerator cpu --experiment_name music_W \
    --regularization_type W --K 10 --lambda_reg 0.01 --batch_size 64 --n_epoch 1000 --seed $seed &

    wait
done
