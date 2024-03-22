gpu_id=0
for gam in 0.01
do
for lam in 0.0001
do
for type in W F no
do
for seed in 0
do
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --accelerator gpu \
--experiment_name "cifar-syn-test_${type}_lambda${lam}_gamma${gam}_model-resnet18" \
--K 10 --M 5 --regularization_type $type --lambda_reg $lam \
--n_epoch 30 --classifier_NN torchvision.models.resnet18 --use_pretrained --dataset cifar10 \
--annotator_type synthetic --num_workers 6 --gamma $gam --seed $seed \
--plot_confusion_matrices &
gpu_id=$((gpu_id+1))
done
done
done
done