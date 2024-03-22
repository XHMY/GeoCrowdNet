# Unofficial Implementation of "Deep learning from crowdsourced labels: Coupled cross-entropy minimization, identifiability, and regularization" (https://arxiv.org/abs/2306.03288)

The official implementation is available [here](https://github.com/shahanaibrahimosu/end-to-end-crowdsourcing).

## Installation

Please install the latest version of PyTorch before running the code.

```bash
pip install -r requirements.txt
```
## Obtaining the data

- CIFAR-10: run the python script at `prepare_data/get_cifar_data.py`
- Food11: download the dataset from [here](https://www.kaggle.com/trolukovich/food11-image-dataset) and extract it to `data/food11`, then run the python script at `prepare_data/get_food11.py`
- Music: from the author's [repository](https://github.com/shahanaibrahimosu/end-to-end-crowdsourcing/tree/master/data/Music)
- MNIST: from the repository [here](https://github.com/sebastian-lapuschkin/lrp_toolbox/tree/master/data/MNIST)

I also upload all the datasets in Box, you can download it [here](https://oregonstate.box.com/s/w5q4w3eihcjn3bzvcczj1j2wxurqzqbw).

## Training

To train the model, run the following command:

```bash
# MUSIC
python main.py --accelerator cpu --dataset music \
--experiment_name "music_W_lambda0.01" \
--regularization_type $type --K 10 --M 44 \
--lambda_reg 0.01 --batch_size 128 --n_epoch 1000 --seed 0 \
--plot_confusion_matrices

# CIFAR-10
CUDA_VISIBLE_DEVICES=0 python main.py --accelerator gpu \
--experiment_name "cifar-syn_F_lambda0.0001_gamma0.01_model-resnet9" \
--K 10 --M 5 --regularization_type F --lambda_reg 0.0001 \
--n_epoch 30 --classifier_NN resnet9 --dataset cifar10 \
--annotator_type synthetic --num_workers 6 --gamma 0.01 --seed 0

# FOOD11
CUDA_VISIBLE_DEVICES=0 python main.py --accelerator gpu \
--experiment_name "food11_F_lambda0.0001_gamma0.01_model-resnet50" \
--K 10 --M 5 --regularization_type F --lambda_reg 0.0001 \
--n_epoch 30 --classifier_NN torchvision.models.resnet50 --use_pretrained --dataset food11 \
--annotator_type synthetic --num_workers 6 --gamma 0.01 --seed 0
```

You can add the `--plot_confusion_matrices` flag to plot the confusion matrices of the annotators.