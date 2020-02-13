from ...Utils.model import Model
from ...Utils import loader
import numpy as np
from ...Benchmarks.OAS_dataset import OAS_data_loader

# -------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_RNN_classifier(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(LSTM_RNN_classifier, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.dropout_rate = 0.5
        if 'hidden_dim' not in para_dict:
            self.hidden_dim = 40
        if 'hidden_layer_num' not in para_dict:
            self.hidden_layer_num = 3

    def net_init(self):
        self.lstm = nn.LSTM(21, self.para_dict['hidden_dim'], batch_first=False,
                            num_layers=self.para_dict['hidden_layer_num'], dropout=self.para_dict['dropout_rate'])
        self.fc = nn.Linear(self.para_dict['hidden_dim'], 2)

    def forward(self, Xs):
        batch_size = len(Xs)
        Xs_len = []
        Xs = np.array(Xs)
        for a in Xs:
            m = np.where(a == 0)[0]
            if not list(m):
                Xs_len.append(Xs.shape[1])
            else:
                Xs_len.append(m[0])
        a = loader.encode_data(np.array(Xs, dtype=int), aa_list='0ACDEFGHIKLMNPQRSTVWY')
        X = torch.FloatTensor(a)
        # X = X.permute(1, 0, 2)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, torch.tensor(Xs_len), batch_first=True, enforce_sorted=False)
        X, _ = self.lstm(X)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        out = out.permute(1, 0, 2)
        out = out[-1, :, :].reshape(batch_size, -1)  # use the last output as input for next layer
        out = torch.sigmoid(self.fc(out))

        return out


# --------------------------------------------------------------
if __name__ == '__main__':
    para_dict = {'num_samples': 1000,
                 'seq_len': 10,
                 'batch_size': 10,
                 'model_name': 'LSTM_Model',
                 'optim_name': 'Adam',
                 'epoch': 50,
                 'learning_rate': 0.001,
                 'step_size': 5,
                 'hidden_dim': 40,
                 'hidden_layer_num': 3,
                 'dropout_rate': 0.5}

    # data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'])
    # data = loader.encode_data(data)
    # train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'])
    # model = LSTM_RNN_classifier(para_dict)

    train_data_human = OAS_data_loader.OAS_data_loader(
        index_file='./antibody-in-pytorch/Benchmarks/OAS_dataset/data/OAS_meta_info.txt', output_field='Species',
        input_type='full_length', species_type=['human', 'mouse'], num_files=50, gapped=False)
    train_x = [x for x, y in train_data_human]
    para_dict['seq_len'] = len(max(train_x, key=len))
    train_y = [1 if y == 'human' else 0 for x, y in train_data_human]
    # print(train_y)
    # print(len(train_x), len(train_y))
    train_x = OAS_data_loader.encode_index(data=train_x, aa_list='ACDEFGHIKLMNPQRSTVWY', pad=True)
    # train_x = loader.encode_data(np.array(train_x), aa_list='0ACDEFGHIKLMNPQRSTVWY')
    train_loader, test_loader = loader.train_test_loader(np.array(train_x), np.array(train_y), test_size=0.3,
                                                         batch_size=para_dict['batch_size'], sample=True, random_state=100)

    model = LSTM_RNN_classifier(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    mat, acc, mcc = model.evaluate(output, labels)

    print(para_dict)
