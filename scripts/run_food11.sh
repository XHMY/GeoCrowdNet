for seed in 1 2 3 4 5
do
for gam in 0.01 0.2 0.3
do
for lam in 0.0001 0.001 0.01
do
for type in W F no
do
CUDA_VISIBLE_DEVICES=0 python main.py --accelerator gpu --experiment_name "food-syn_${type}_lambda${lam}_gamma${gam}" \
--K 10 --M 5 --regularization_type $type --lambda_reg $lam \
--n_epoch 30 --dataset cifar10 --batch_size 64 \
--classifier_NN torchvision.models.resnet34 --use_pretrained \
--annotator_type synthetic --num_workers 6 --gamma $gam --seed $seed
done
done
done
done
