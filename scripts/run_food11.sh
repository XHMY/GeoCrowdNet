for gam in 0.01 0.2 0.3
do
for lam in 0.0001 0.001 0.01
do
for type in W F no
do
for model in vgg16_bn resnet18 # swin_v2_t resnet50
do
for seed in 0
do
CUDA_VISIBLE_DEVICES=$((seed-0)) python main.py --accelerator gpu \
--experiment_name "food-syn_${type}_lambda${lam}_gamma${gam}_${model}" \
--K 11 --M 5 --regularization_type $type --lambda_reg $lam \
--n_epoch 30 --dataset food11 --batch_size 256 \
--classifier_NN "torchvision.models.${model}" --use_pretrained \
--annotator_type synthetic --num_workers 6 --gamma $gam --seed $seed &
done
wait
done
done
done
done