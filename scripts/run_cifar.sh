for seed in 2 3 4 5
do
for gam in 0.01 0.2 0.3
do
for lam in 0.0001 0.001 0.01
do
for type in W
do
CUDA_VISIBLE_DEVICES=1 python main.py --accelerator gpu --experiment_name "cifar-syn_${type}_lambda${lam}_gamma${gam}" \
--K 10 --M 5 --regularization_type $type --lambda_reg $lam \
--n_epoch 30 --classifier_NN resnet9 --dataset cifar10 \
--annotator_type synthetic --num_workers 8 --gamma $gam --seed $seed
done
done
done
done

for seed in 2 3 4 5
do
for gam in 0.01 0.2 0.3
do
for lam in 0.0001 0.001 0.01
do
for type in F
do
CUDA_VISIBLE_DEVICES=2 python main.py --accelerator gpu --experiment_name "cifar-syn_${type}_lambda${lam}_gamma${gam}" \
--K 10 --M 5 --regularization_type $type --lambda_reg $lam \
--n_epoch 30 --classifier_NN resnet9 --dataset cifar10 \
--annotator_type synthetic --num_workers 8 --gamma $gam --seed $seed
done
done
done
done

for seed in 2 3 4 5
do
for gam in 0.01 0.2 0.3
do
CUDA_VISIBLE_DEVICES=3 python main.py --accelerator gpu --experiment_name "cifar-syn_no_gamma${gam}" \
--K 10 --M 5 --regularization_type no \
--n_epoch 30 --classifier_NN resnet9 --dataset cifar10 \
--annotator_type synthetic --num_workers 8 --gamma $gam --seed $seed
done
done