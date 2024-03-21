for lam in 0.0001 0.001 0.01
do
for seed in 0 1 2 3 4
do
for type in W F no
do
python main.py --accelerator cpu \
--experiment_name "music_${type}_lambda${lam}" \
--regularization_type $type --K 10 --M 44 \
--lambda_reg $lam --batch_size 64 --n_epoch 1000 --seed $seed &
done
done
wait
done
