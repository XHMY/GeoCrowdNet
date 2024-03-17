import argparse
import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from crowdsourcing_data_module import CrowdsourcingDataModule
from model import GeoCrowdNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, help='No of annotators', default=5)
    parser.add_argument('--K', type=int, help='No of classes', default=3)
    parser.add_argument('--N', type=int, help='No of data samples (synthetic data)', default=10000)
    parser.add_argument('--R', type=int, help='Dimension of data samples (synthetic data)', default=5)
    parser.add_argument('--annotator_label_pattern', type=str,
                        help='random or correlated or per-sample-budget or per-annotator-budget',
                        default='per-sample-budget')
    parser.add_argument('--l', type=int, help='number of annotations per sample or number of samples per annotators',
                        default=1)
    parser.add_argument('--p', type=float, help='prob. that an annotator label a sample', default=0.2)
    parser.add_argument('--conf_mat_type', type=str, help='separable, random, or diagonally-dominant,' \
                                                          'hammer-spammer, classwise-hammer-spammer, pairwise-flipper',
                        default='separable-and-uniform')
    parser.add_argument('--gamma', type=float, help='hammer probability in hammer-spammer type', default=0.01)
    parser.add_argument('--dataset', type=str, help='synthetic or cifar10 or mnist', default='labelme')
    parser.add_argument('--annotator_type', type=str,
                        help='synthetic, machine-classifier, good-bad-annotator-mix or real', default='real')
    parser.add_argument('--good_bad_annotator_ratio', type=float,
                        help='ratio of good:bad annotators for good-bad-annotator-mix type ', default=0.1)
    parser.add_argument('--flag_preload_annotations', type=bool,
                        help='True or False (if True, load annotations from file, otherwise generate annotations', \
                        default=True)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--n_trials', type=int, help='No of trials', default=5)
    parser.add_argument('--flag_hyperparameter_tuning', type=bool, help='True or False', default=True)
    parser.add_argument('--proposed_init_type', type=str,
                        help='close_to_identity or mle_based or deviation_from_identity', default='close_to_identity')
    parser.add_argument('--proposed_projection_type', type=str,
                        help='simplex_projection or softmax or sigmoid_projection', default='simplex_projection')
    parser.add_argument('--classifier_NN', type=str, help='resnet9 or resnet18 or resnet34', default='resnet9')

    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--batch_size', type=int, help='Batch Size', default=128)
    parser.add_argument('--n_epoch', type=int, help='Number of Epochs', default=200)
    parser.add_argument('--n_epoch_maxmig', type=int, help='Number of Epochs for Maxmig', default=20)
    parser.add_argument('--coeff_label_smoothing', type=float, help='label smoothing coefficient', default=0)
    parser.add_argument('--log_folder', type=str, help='log folder path', default='results/labelme_real/')
    parser.add_argument('--n_epoch_mv', type=int, help='majority_voting_init_epochs', default=20)

    # New arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--accelerator', type=str, default='gpu', help='cpu, gpu, tpu')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default='geo_crowd_net', help='Experiment name')
    parser.add_argument('--log_every_n_steps', type=int, default=4, help='Log every N steps')
    parser.add_argument('--regularization_type', type=str, default='F', help='Regularization type (F or W)')
    parser.add_argument('--lambda_reg', type=float, default=0.2, help='Regularization strength')
    parser.add_argument('--plain', action='store_true', default=False, help='Use plain model (no confusion matrices)')

    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed)

    dm = CrowdsourcingDataModule(dataset_name="music", data_dir="data", logger=logging.getLogger(),
                                 batch_size=args.batch_size, num_workers=args.num_workers, args=args)
    dm.setup()

    model = GeoCrowdNet(input_dim=124, num_classes=10, num_annotators=44,
                        # init_method='mle_based', annotations_list=dm.full_dataset.annotations_list_maxmig,
                        init_method='identity',
                        regularization_type=args.regularization_type, lambda_reg=args.lambda_reg, args=args)

    # early_stop_callback = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, save_top_k=1, monitor='val_loss', mode='min')
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)

    trainer = pl.Trainer(max_epochs=args.n_epoch, accelerator=args.accelerator,
                         callbacks=[checkpoint_callback], # early_stop_callback
                         logger=logger, log_every_n_steps=args.log_every_n_steps)

    trainer.fit(model, dm)

    # Load the best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    best_model = GeoCrowdNet.load_from_checkpoint(best_model_path, input_dim=124, num_classes=10, num_annotators=44,
                                                  regularization_type=args.regularization_type,
                                                  init_method='identity',
                                                  lambda_reg=args.lambda_reg, args=args)

    # Test the best model
    results = trainer.test(best_model, datamodule=dm)
    with open(f'logs/results.txt', 'a') as f:
        f.write(f"{args.experiment_name},{args.seed},{results[0]['test_accuracy']}\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)
