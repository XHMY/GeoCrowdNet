for lam in 0.0001 0.001 0.01
do
for type in W F
do
for seed in 0 1 2 3 # 4
do
for gam in 0.01 0.2 0.3
do
CUDA_VISIBLE_DEVICES=0 python main.py --accelerator gpu \
--experiment_name "cifar-b32-M32-syn_${type}_lambda${lam}_gamma${gam}_model-resnet18" \
--K 10 --M 32 --batch_size 32 --regularization_type $type --lambda_reg $lam \
--n_epoch 30 --classifier_NN torchvision.models.resnet18 --use_pretrained --dataset cifar10 \
--annotator_type synthetic --num_workers 6 --gamma $gam --seed $seed &
done

for gam in 0.01 0.2 0.3
do
CUDA_VISIBLE_DEVICES=1 python main.py --accelerator gpu \
--experiment_name "cifar-b32-M32-syn_${type}_lambda${lam}_gamma${gam}_model-resnet9" \
--K 10 --M 32 --batch_size 32 --regularization_type $type --lambda_reg $lam \
--n_epoch 30 --classifier_NN resnet9 --dataset cifar10 \
--annotator_type synthetic --num_workers 6 --gamma $gam --seed $seed &
done
wait
done
done
done

for seed in 0 1 2 3 # 4
do
for gam in 0.01 0.2 0.3
do
CUDA_VISIBLE_DEVICES=0 python main.py --accelerator gpu \
--experiment_name "cifar-b32-M32-syn_no_gamma${gam}_model-resnet18" \
--K 10 --M 32 --batch_size 32 --regularization_type $type \
--n_epoch 30 --classifier_NN torchvision.models.resnet18 --use_pretrained --dataset cifar10 \
--annotator_type synthetic --num_workers 6 --gamma $gam --seed $seed &
done

for gam in 0.01 0.2 0.3
do
CUDA_VISIBLE_DEVICES=1 python main.py --accelerator gpu \
--experiment_name "cifar-b32-M32-syn_no_gamma${gam}_model-resnet9" \
--K 10 --M 32 --batch_size 32 --regularization_type $type \
--n_epoch 30 --classifier_NN resnet9 --dataset cifar10 \
--annotator_type synthetic --num_workers 6 --gamma $gam --seed $seed &
done
wait
done


