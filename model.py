import os
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision
from helpers.arch import ResNet9, Lenet
from helpers.visualize import plot_confusion_matrices


def confusion_matrix_init_mle_based(annotations, M, K):
    # MLE initialization for confusion matrices
    # annotations: list of annotations from all annotators
    # M: number of annotators
    # K: number of classes

    confusion_matrices = torch.zeros((M, K, K))

    for m in range(M):
        for j in range(K):
            for k in range(K):
                numerator = torch.sum((annotations == k) * (annotations[:, m] == j))
                denominator = torch.sum((annotations[:, m] == j).float())
                if denominator > 0:
                    confusion_matrices[m, j, k] = numerator / denominator

    return confusion_matrices


class GeoCrowdNet(pl.LightningModule):
    def __init__(self, input_dim, num_classes, num_annotators, regularization_type='F', lambda_reg=0.01,
                 init_method='identity', args=None, annotations_list=None):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        if self.args.classifier_NN == 'fcnn_dropout_batchnorm':
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
                nn.Softmax(dim=1)
            )
        elif self.args.classifier_NN == 'resnet9':
            self.classifier = ResNet9(num_classes=num_classes)
        elif self.args.classifier_NN == 'lenet':
            self.classifier = Lenet(num_classes=num_classes)
        elif self.args.classifier_NN.startswith('torchvision.models'):
            self.classifier = eval(self.args.classifier_NN)(pretrained=self.args.use_pretrained)
            if "resnet" in self.args.classifier_NN:
                self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
            elif "swin" in self.args.classifier_NN:
                self.classifier.head = nn.Linear(self.classifier.head.in_features, num_classes)
            elif "vgg" in self.args.classifier_NN:
                self.classifier.classifier[6] = nn.Linear(self.classifier.classifier[6].in_features, num_classes)
        else:
            raise ValueError(f"Invalid fnet_type: {self.args.classifier_NN}")

        if init_method == 'identity':
            self.confusion_matrices = nn.ParameterList(
                [nn.Parameter(torch.eye(num_classes)) for _ in range(num_annotators)])
        elif init_method == 'mle_based':
            assert annotations_list is not None, "Annotations must be provided for MLE-based initialization"
            mle_confusion_matrices = confusion_matrix_init_mle_based(annotations_list, num_annotators, num_classes)
            self.confusion_matrices = nn.ParameterList([nn.Parameter(cm) for cm in mle_confusion_matrices])
        elif init_method == 'deviation_from_identity':
            self.confusion_matrices = nn.ParameterList(
                [nn.Parameter(torch.eye(num_classes) + 0.1 * torch.randn(num_classes, num_classes)) for _ in
                 range(num_annotators)])
        else:
            raise ValueError(f"Invalid initialization method: {init_method}")

    def forward(self, x):
        f_outputs = self.classifier(x)
        # A = torch.stack([F.softmax(cm, dim=1) for cm in self.confusion_matrices])
        A = torch.stack([cm for cm in self.confusion_matrices])
        y = torch.einsum('ij, bkj -> ibk', f_outputs, A)
        return f_outputs, y, A

    def plot_confusion_matrices(self, output_dir=None):
        os.makedirs(output_dir, exist_ok=True)
        plot_confusion_matrices([F.softmax(cm.cpu().detach(), dim=0).numpy() for cm in self.confusion_matrices], join(output_dir, f"cm_epoch{self.current_epoch}_"))

    def on_validation_epoch_end(self) -> None:
        if self.args.plot_confusion_matrices:
            self.plot_confusion_matrices(output_dir=join("figures", self.logger.name))

    def training_step(self, batch, batch_idx):
        x, annotations, annot_onehot, annot_mask, annot_list, y = batch

        f_outputs, y_preds, A = self(x)

        if self.args.plain:
            loss = F.cross_entropy(f_outputs, y)
        else:
            loss = 0
            for m in range(self.num_annotators):
                mask = annot_mask[:, m] != 0  # Handle missing labels
                if torch.any(mask):
                    loss += F.cross_entropy(y_preds[mask, m], annot_onehot[mask, m])

            loss /= self.num_annotators

        if self.regularization_type == 'F':
            F_matrix = f_outputs
            reg_term = torch.logdet(torch.matmul(F_matrix.T, F_matrix))
            # check if reg_term is nan
            if torch.isnan(reg_term) or torch.isinf(reg_term):
                reg_term = 0
                print('reg_term is nan or inf')
        elif self.regularization_type == 'W':
            W_matrix = F.softmax(torch.stack([cm for cm in self.confusion_matrices]), dim=1)
            W_matrix = W_matrix.view(self.num_annotators * self.num_classes, self.num_classes)
            reg_term = torch.logdet(torch.matmul(W_matrix.T, W_matrix))
        else:
            reg_term = 0

        loss -= self.lambda_reg * reg_term

        self.log('train_loss', loss)
        self.train_accuracy(torch.argmax(f_outputs, dim=1), y)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, annotations, annot_onehot, annot_mask, annot_list, y = batch
        f_outputs, y_preds, A = self(x)
        loss = F.cross_entropy(f_outputs, y)
        self.log('val_loss', loss)
        self.val_accuracy(torch.argmax(f_outputs, dim=1), y)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        f_outputs, y_preds, A = self(x)
        self.test_accuracy(torch.argmax(f_outputs, dim=1), y)
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        return optimizer
