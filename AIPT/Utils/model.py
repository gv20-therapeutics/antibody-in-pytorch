import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score

from . import loader

warnings.filterwarnings("ignore")

class CrossEntropy():
    def __init__(self, *args, **kwargs):
        super(CrossEntropy, self).__init__(*args, **kwargs)

    def __call__(self, para_dict, outputs, targets):
        l = nn.CrossEntropyLoss()
        loss = l(outputs, torch.tensor(targets).type(torch.long))
        return loss

class Model(nn.Module):

    def __init__(self, para_dict, *args, **kwargs):
        super(Model, self).__init__()

        self.para_dict = para_dict

        if 'work_path' not in para_dict:
            self.para_dict['work_path'] = os.path.join(os.getcwd(), 'work')
        if 'model_name' not in para_dict:
            self.para_dict['model_name'] = 'Model'
        if 'seq_len' not in para_dict:
            self.para_dict['seq_len'] = 10
        if 'epoch' not in para_dict:
            self.para_dict['epoch'] = 50
        if 'batch_size' not in para_dict:
            self.para_dict['batch_size'] = 20
        if 'step_size' not in para_dict:
            self.para_dict['step_size'] = 5
        if 'learning_rate' not in para_dict:
            self.para_dict['learning_rate'] = 0.01
        if 'optim_name' not in para_dict:
            self.para_dict['optim_name'] = 'Adam'

        self.work_path = para_dict['work_path']
        self.model_path = os.path.join(self.work_path, self.para_dict['model_name'] + '_' + str(
            self.para_dict['batch_size']))
        self.save_path = os.path.join(self.model_path, 'model')

        if not os.path.exists(self.work_path):
            os.mkdir(self.work_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if self.load_param() is None:
            self.save_param()

    def net_init(self):

        self.fc = nn.Linear(20 * self.para_dict['seq_len'], 2)

    def forward(self, x):

        x = x.view(x.shape[0], 20 * self.para_dict['seq_len'])
        return x

    def objective(self):
        return CrossEntropy()

    def optimizers(self):

        if self.para_dict['optim_name'] == 'Adam':
            return optim.Adam(self.parameters(), lr=self.para_dict['learning_rate'])

        elif self.para_dict['optim_name'] == 'RMSprop':
            return optim.RMSprop(self.parameters(), lr=self.para_dict['learning_rate'])

        elif self.para_dict['optim_name'] == 'SGD':
            return optim.SGD(self.parameters(), lr=self.para_dict['learning_rate'])

    def fit(self, data_loader):

        self.net_init()
        saved_epoch = self.load_model()

        self.train()
        optimizer = self.optimizers()
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.para_dict['step_size'],
        #                                       gamma = 0.5)
        for e in range(saved_epoch, self.para_dict['epoch']):
            print('Epoch %d: ' % (e + 1), end='')
            total_loss = 0
            for input in data_loader:
                features, labels = input
                logps = self.forward(features)
                loss = self.objective()
                loss = loss(self.para_dict, logps, labels)  # torch.tensor(labels).type(torch.long)
                total_loss += loss
                # outputs_train.append(logps.detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # scheduler.step()

            self.save_model('Epoch_' + str(e + 1), self.state_dict())
            print('Loss=%.3f' % (total_loss))

    # def predict(self, data_loader):
    #
    #     self.eval()
    #     test_loss = 0
    #     all_outputs = []
    #     with torch.no_grad():
    #         for data in data_loader:
    #             inputs, _ = data
    #             outputs = self.forward(inputs)
    #             temp = []
    #             for i in range(len(outputs)):
    #                 temp.append(outputs[i].detach().numpy())
    #             all_outputs.append(temp)
    #         return np.vstack(all_outputs)

    def predict(self, data_loader):

        self.eval()
        test_loss = 0
        all_outputs = []
        labels_test = []
        with torch.no_grad():
            for data in data_loader:
                # print(data)
                inputs, _ = data
                outputs = self.forward(inputs)
                all_outputs.append(outputs.detach().numpy())
                # labels_test.append(np.array(l))

            return np.vstack(all_outputs)

    # def evaluate(self, outputs, labels):
    #     outputs = np.array(outputs).T
    #     labels = np.array(labels).T
    #     num_tasks = labels.shape[0]
    #     for i in range(num_tasks):
    #         y_pred = []
    #         for a in outputs[i]:
    #             y_pred.append(np.argmax(a))
    #         y_true = np.array(labels[i]).flatten()
    #         y_pred = np.array(y_pred)
    #         mat = confusion_matrix(y_true, y_pred)
    #         acc = accuracy_score(y_true, y_pred)
    #         mcc = matthews_corrcoef(y_true, y_pred)
    #
    #         print('Confusion matrix: ')
    #         print(mat)
    #         print('Accuracy = %.3f, MCC = %.3f' % (acc, mcc))
    #
    #     return mat, acc, mcc

    def evaluate(self, outputs, labels):

        # outputs = np.array(outputs).T
        # labels = np.array(labels).T
        y_pred = []
        for a in outputs:
            y_pred.append(np.argmax(a))
        y_true = np.array(labels).flatten()
        y_pred = np.array(y_pred)
        mat = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        print('Confusion matrix: ')
        print(mat)
        print('Accuracy = %.3f, MCC = %.3f' % (acc, mcc))

        return mat, acc, mcc

    def save_model(self, filename, model):
        torch.save(model, os.path.join(self.save_path, filename))

    def load_model(self):

        for e in range(self.para_dict['epoch'], 0, -1):
            if os.path.isfile(os.path.join(self.save_path, 'Epoch_' + str(e))):
                # print(os.path.join(self.save_path, 'Epoch_' + str(e)))
                self.load_state_dict(torch.load(os.path.join(self.save_path, 'Epoch_' + str(e))))
                return e
        return 0

    def save_param(self, path=None):
        if path == None:
            filepath = os.path.join(self.model_path, 'train_parameters.json')
        else:
            filepath = os.path.join(path, 'train_parameters.json')
        with open(filepath, 'w') as f:
            json.dump(self.para_dict, f, indent=2)

    def load_param(self, path=None):
        if path == None:
            filepath = os.path.join(self.model_path, 'train_parameters.json')
        else:
            filepath = os.path.join(path, 'train_parameters.json')
        if os.path.exists(filepath):
            return json.load(open(filepath, 'r'))
        return None

    def collate_fn(batch):
        return batch, [x for seq in batch for x in seq]


def test():
    para_dict = {'num_samples': 1000,
                 'seq_len': 20,
                 'batch_size': 20,
                 'model_name': 'Model',
                 'optim_name': 'Adam',
                 'step_size': 10,
                 'epoch': 2,
                 'learning_rate': 0.01}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'])
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=20)

    print('Parameters are', para_dict)
    model = Model(para_dict)
    print('Training...')
    model.fit(train_loader)
    print('Testing...')
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    model.evaluate(output, labels)

if __name__ == '__main__':
    test()