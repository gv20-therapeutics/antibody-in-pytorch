import json
import math
import os

import AIPT.Utils.Dev.dev_utils as dev_utils
import AIPT.Utils.logging as aipt_logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from AIPT.Utils import loader
from AIPT.Utils.metrics import binary_classification_metrics
from AIPT.Utils.model import Model
from AIPT.Utils.plotting import plot_confusion_matrix, plot_to_image
from torch.utils.tensorboard import SummaryWriter

DEBUG_MODE = True
if DEBUG_MODE:
    # print last modified time upon import, useful in Jupyter to make sure most recent version is imported.
    dev_utils.get_mod_time(__file__, verbose=True)


class CNN(Model):
    def __init__(self, para_dict, embedding_fn, *args, **kwargs):
        '''
        Initialize CNN parameters.

        Args:
            para_dict (dict of str: various types): CNN parameter dict, includes hyperparameters such as
            "learning_rate", as well as run parameters such as "work_path".
            embedding_fn (torch.nn.Embedding): takes AA sequence (str) aa_seq and outputs embedding of dimension
            para_dict[embedding_dim]
        '''

        super(CNN, self).__init__(para_dict, *args, **kwargs)  # todo: what's the point of this line?

        self.embedding_fn = embedding_fn
        self.para_dict.setdefault('batch_size', 100)
        self.num_classes = len(self.para_dict['classes'])

        # architecture spec
        self.para_dict.setdefault('embedding_dim', 15)
        self.para_dict.setdefault('dropout_rate', 0.4)
        self.para_dict.setdefault('conv1_n_filters', 8)
        self.para_dict.setdefault('conv2_n_filters', 16)
        self.para_dict.setdefault('conv1_filter_size', (self.para_dict['embedding_dim'], 2))
        self.para_dict.setdefault('conv2_filter_size', (1, 2))
        self.para_dict.setdefault('max_pool_filter_size', (1, 2))
        self.para_dict.setdefault('fc_hidden_dim', 10)
        self.para_dict.setdefault('stride', 1)
        self.para_dict.setdefault('padding', 0)

        # logging
        self.para_dict.setdefault('run_name', aipt_logging.today())
        self.para_dict.setdefault('log_dir', 'logs')

        self.writer = SummaryWriter(log_dir=self.para_dict['log_dir'])  # TensorBoard writer

        # best/final model paths
        self.best_model_fn = 'best'
        self.best_model_path = os.path.join(self.save_path, self.best_model_fn)

        self.final_model_fn = 'final'
        self.final_model_path = os.path.join(self.save_path, self.final_model_fn)

        # GPU
        # todo: code has not been tested with GPU
        self.para_dict.setdefault('GPU', False)
        if self.para_dict['GPU']:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')


    def net_init(self):
        '''
        Defines network layers.
        '''


        def get_width_after_conv(width, filter_size, stride, padding=0):
            '''
            Calculates output dimension of 1D input following application of a 1D convolutional filter.

            Args:
                width (int): input size
                filter_size (int): filter size
                stride (int): convolutional stride
                padding (int): convolutional padding

            Returns (int): Output size after applying convolutional filter.

            '''
            return (width - filter_size + 2 * padding) // stride + 1


        start_width = self.para_dict['seq_len']
        self.conv1 = nn.Conv2d(in_channels=1,  # todo: support gapped?
                               out_channels=self.para_dict['conv1_n_filters'],
                               kernel_size=self.para_dict['conv1_filter_size'], stride=self.para_dict['stride'],
                               padding=self.para_dict['padding'])
        width = get_width_after_conv(start_width, self.para_dict['conv1_filter_size'][1], self.para_dict['stride'],
                                     padding=self.para_dict['padding'])
        self.pool1 = nn.MaxPool2d(kernel_size=self.para_dict['max_pool_filter_size'],
                                  stride=self.para_dict['stride'])
        width = get_width_after_conv(width, self.para_dict['max_pool_filter_size'][1], self.para_dict['stride'])
        self.conv2 = nn.Conv2d(in_channels=self.para_dict['conv1_n_filters'],
                               out_channels=self.para_dict['conv2_n_filters'],
                               kernel_size=self.para_dict['conv2_filter_size'], stride=self.para_dict['stride'],
                               padding=self.para_dict['padding'])
        width = get_width_after_conv(width, self.para_dict['conv2_filter_size'][1], self.para_dict['stride'],
                                     padding=self.para_dict['padding'])
        self.pool2 = nn.MaxPool2d(kernel_size=self.para_dict['max_pool_filter_size'],
                                  stride=self.para_dict['stride'])
        width = get_width_after_conv(width, self.para_dict['max_pool_filter_size'][1], self.para_dict['stride'])
        in_features = width * self.para_dict['conv2_n_filters']
        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=self.para_dict['fc_hidden_dim'])
        self.logits = nn.Linear(in_features=self.para_dict['fc_hidden_dim'],
                                out_features=self.num_classes)
        self.dropout = nn.Dropout(p=self.para_dict['dropout_rate'])

        if self.para_dict['GPU']:
            self.cuda()


    def forward(self, Xs):
        '''
        Forward pass during training.

        Args:
            Xs (torch.Tensor, shape=[batch_size, seq_length]): Network input.

        Returns (torch.Tensor, shape=[batch_size, num_classes]): Output of network on `Xs`.

        '''
        batch_size = len(Xs)

        if self.para_dict['GPU']:
            X = Xs.cuda()

        out = self.embedding_fn(Xs)
        out = out.permute(0, 2, 1)
        out = out.unsqueeze(1)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = torch.reshape(out, [batch_size, -1])
        out = self.fc1(out)
        out = self.dropout(F.relu(out))
        out = self.logits(out)

        return out


    def predict(self, data_loader):
        '''
        Model output during inference.

        Args:
            data_loader (torch.utils.data.DataLoader): data loader

        Returns (tuple of (np.array, np.array, float)): Tuple of (outputs, labels, loss)

        '''
        self.eval()
        all_outputs = []
        labels_test = []
        loss_func = self.objective()
        total_loss = 0
        with torch.no_grad():
            for data in data_loader:
                inputs, label = data
                outputs = self.forward(inputs)
                total_loss += loss_func(self.para_dict, outputs, torch.tensor(label).type(torch.long))
                outputs = torch.nn.Softmax()(outputs)
                labels_test.append(label)

                if self.para_dict['GPU']:
                    all_outputs.append(outputs.cpu().detach().numpy())
                else:
                    all_outputs.append(outputs.detach().numpy())

            return np.vstack(all_outputs), np.hstack(labels_test), total_loss


    def fit(self, train_loader, test_loader=None):
        '''
        Train model.

        Args:
            train_loader (torch.utils.data.DataLoader): train data
            test_loader (torch.utils.data.DataLoader): test data

        Returns: Dictionary containing evaluation metrics. If `test_loader` specified, test metrics are also included.

            Example dictionary structure:
                {
                    'train': {
                               'cm': train_cm,
                               'acc': train_acc,
                               'mcc': train_mcc
                             },
                    'test':  {
                                 'cm': test_cm,
                                 'acc': test_acc,
                                 'mcc': test_acc,
                                 'best_mcc': best_test_mcc,
                                 'best_epoch': best_epoch
                             }
                }

        '''
        self.net_init()
        saved_epoch = self.load_model()
        if saved_epoch:
            print('Found saved model from: Epoch', saved_epoch)
        else:
            saved_epoch = 0
            print('No saved model found.')
        print()

        self.train()
        optimizer = self.optimizers()
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.para_dict['step_size'], gamma=0.5 ** (
        # self.para_dict['epoch'] / self.para_dict['step_size']))

        loss_func = self.objective()
        out_metrics = {}

        best_test_mcc = -math.inf

        for epoch in range(saved_epoch, self.para_dict['epoch'] + 1):
            total_loss = 0
            labels_train = []
            for input in train_loader:
                features, labels = input
                labels_train.append(labels)
                logps = self.forward(features)
                if self.para_dict['GPU']:
                    loss = loss_func(self.para_dict, logps, labels.type(torch.long).cuda())
                else:
                    loss = loss_func(self.para_dict, logps, torch.tensor(labels).type(torch.long))

                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # scheduler.step()

            if epoch == saved_epoch or epoch % 10 == 0:  # always do initial eval, otherwise evaluate periodically
                sep = 50 * '='
                print(sep)
                print(f'EPOCH {epoch}:')
                print()
                if epoch == 0 or epoch != saved_epoch:
                    self.save_model(f'Epoch_{epoch}', self.state_dict())  # todo: consider saving only best model

                train_metrics = self.evaluate(train_loader, data_name='Train', log_epoch=epoch)

                if test_loader is not None:
                    test_metrics = self.evaluate(test_loader, data_name='Test', log_epoch=epoch)
                    if test_metrics['mcc'] > best_test_mcc:  # save best model
                        best_test_mcc, best_epoch = test_metrics['mcc'], epoch
                        self.save_model(f'best_epoch_{epoch}', self.state_dict())
                        self.save_model(self.best_model_fn, self.state_dict())
                    print()
                    print(f'best_epoch={best_epoch};', f'best_test_mcc={best_test_mcc};')  # for AWS metric tracking

                print(sep)
                print('\n')

        self.save_model(self.final_model_fn, self.state_dict())  # todo: consider saving only best model
        out_metrics['train'] = train_metrics

        if test_loader is not None:
            out_metrics['test'] = {
                **test_metrics,
                'best_mcc': best_test_mcc,
                'best_epoch': best_epoch
            }

        return out_metrics


    def evaluate(self, data_loader, data_name='', log_epoch=None):
        '''
        Evaluates model on `data_loader` and optionally logs metrics to TensorBoard.

        Args:
            data_loader (torch.utils.data.DataLoader): Data
            data_name (str): Label to give data loader, ex. 'Train' or 'Test'.
            log_epoch (None or int): If int given, log to TensorBoard under epoch `log_epoch`.

        Returns (dict str: various types): Dict containing evaluation metrics.

        '''
        outputs, labels, loss = self.predict(data_loader)
        if data_name:
            print(f'{data_name.upper()}:')
        num_examples = len(data_loader) * self.para_dict['batch_size']  # exact since we drop last batch if incomplete
        avg_loss = loss / num_examples
        print(f'Total Loss=%.2f' % (loss), f'Average Loss=%.2e' % (avg_loss))

        if log_epoch is not None:
            self.writer.add_scalar(f'Total Loss/{data_name}', loss, log_epoch)  # works with `data_name` = ''
            self.writer.add_scalar(f'Average Loss/{data_name}', avg_loss, log_epoch)  # works with `data_name` = ''

        cm, acc, mcc = self.metrics(outputs, labels, data_name='', log_epoch=log_epoch)
        metrics = {'loss': loss,
                   'cm': cm,  # todo: consider factoring these keys out into constants
                   'acc': acc,
                   'mcc': mcc
                   }
        print(30 * '-')
        return metrics


    def metrics(self, outputs, labels, data_name='', log_epoch=None, verbose=True):
        '''
        Computes evaluation metrics based on outputs and labels, and optionally logs them to TensorBoard.

        Args:
            outputs (sequence of int in {0,1}): Model predicted classes.
            labels (sequence of int in {0,1}): True class labels.
            data_name (str): Name to assign data. Ex. 'Train', 'Test'.
            log_epoch (None or int): If int, metrics are logged to TensorBoard under epoch `log_epoch`.
            verbose (bool): If True, print metrics.

        Returns (tuple of (np.array, float, float)): Tuple of (confusion matrix, accuracy, MCC)

        '''
        cm, acc, mcc = binary_classification_metrics(outputs, labels, verbose=verbose)
        if log_epoch is not None:
            self.writer.add_scalar(f'Accuracy/{data_name}', acc, log_epoch)  # works with `data_name` = ''
            self.writer.add_scalar(f'MCC/{data_name}', mcc, log_epoch)
            cm_plot_test = plot_confusion_matrix(cm, self.para_dict['classes'])
            self.writer.add_image(f'Confusion/{data_name}', plot_to_image(cm_plot_test), log_epoch)
        return cm, acc, mcc


    def load_best(self):
        '''
        Load model with best test MCC. The file at `self.best_model_path` must exist.
        '''
        weights = torch.load(self.best_model_path)
        print(self.best_model_path)
        self.load_state_dict(weights)


    def print_model_params(self):
        '''
        Print model parameters.
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Total number of parameters: %i' % params)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
