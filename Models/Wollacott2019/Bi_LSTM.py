from torch.utils.data import Dataset
from ...Utils.model import Model
import pandas as pd
from ...Benchmarks.OAS_dataset.data import OAS_data_loader
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, matthews_corrcoef, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# true if gapped else false
vocab_o = { True: ['-'] + ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
           False: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']}
aa2id_o = { True: dict(zip(vocab_o[True],  list(range(len(vocab_o[True]))))),
           False: dict(zip(vocab_o[False], list(range(len(vocab_o[False])))))}
id2aa_o = { True: dict(zip(list(range(len(vocab_o[True]))),  vocab_o[True])),
           False: dict(zip(list(range(len(vocab_o[False]))), vocab_o[False]))}

vocab_i = { True: vocab_o[True]  + ['<SOS>', '<EOS>'],
           False: vocab_o[False] + ['<SOS>', '<EOS>']}
aa2id_i = { True: dict(zip(vocab_i[True],  list(range(len(vocab_i[True]))))),
           False: dict(zip(vocab_i[False], list(range(len(vocab_i[False])))))}
id2aa_i = { True: dict(zip(list(range(len(vocab_i[True]))),  vocab_i[True])),
           False: dict(zip(list(range(len(vocab_i[False]))), vocab_i[False]))}


def collate_fn(batch):
    return batch, [x for seq in batch for x in seq]

#--------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

class LSTM_Bi(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(LSTM_Bi, self).__init__(para_dict, *args, **kwargs)

        if 'in_dim' not in para_dict:
            self.in_dim = len(aa2id_i[self.para_dict['gapped']])
        if 'embedding_dim' not in para_dict:
            self.embedding_dim = 64
        if 'hidden_dim' not in para_dict:
            self.hidden_dim = 64
        if 'out_dim' not in para_dict:
            self.out_dim = len(aa2id_o[self.para_dict['gapped']])
        if 'fixed_len' not in para_dict:
            self.fixed_len = True

    def net_init(self):

        # self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(self.para_dict['in_dim'], self.para_dict['embedding_dim'])
        self.lstm_f = nn.LSTM(self.para_dict['embedding_dim'], self.para_dict['hidden_dim'], batch_first=True)
        self.lstm_b = nn.LSTM(self.para_dict['embedding_dim'], self.para_dict['hidden_dim'], batch_first=True)
        self.fc1 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['hidden_dim'])
        self.fc2 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['hidden_dim'])
        self.fc3 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['out_dim'])
        self.fixed_len = self.para_dict['fixed_len']
        self.forward = self.forward_flen if self.fixed_len else self.forward_vlen

    def forward_flen(self, Xs):
        # batch_size = len(Xs)
        batch_size = len(Xs)
        # print(_aa2id)
        _aa2id =  aa2id_i[para_dict['gapped']]

        # pad <EOS> & <SOS>
        Xs_f = [[_aa2id['<SOS>']] + list(seq)[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + list(seq)[::-1][:-1] for seq in Xs]

        # get sequence lengths
        X_len = len(Xs_f[0])

        # list to *.tensor
        Xs_f = torch.tensor(Xs_f)
        Xs_b = torch.tensor(Xs_b)

        # print(Xs_f)
        # embedding
        Xs_f = self.word_embeddings(Xs_f.type(dtype=torch.LongTensor))
        Xs_b = self.word_embeddings(Xs_b.type(dtype=torch.LongTensor))

        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, batch_size, self.para_dict['hidden_dim']),
                          torch.zeros(1, batch_size, self.para_dict['hidden_dim']))
        ini_hc_state_b = (torch.zeros(1, batch_size, self.para_dict['hidden_dim']),
                          torch.zeros(1, batch_size, self.para_dict['hidden_dim']))

        # lstm
        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)

        # flatten forward-lstm output
        lstm_out_f = lstm_out_f.reshape(-1, self.para_dict['hidden_dim'])

        # flatten backward-lstm output
        idx_b = torch.tensor(list(range(X_len))[::-1])
        lstm_out_b = torch.index_select(lstm_out_b, 1, idx_b)
        lstm_out_b = lstm_out_b.reshape(-1, self.para_dict['hidden_dim'])

        lstm_out_valid = lstm_out_f + lstm_out_b

        # lstm hidden state to output space
        out = F.relu(self.fc1(lstm_out_valid))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # print(out.shape)

        # compute scores
        scores = F.log_softmax(out, dim=1)

        return scores

    def forward_vlen(self, Xs):
        batch_size = len(Xs)
        _aa2id = aa2id_i[para_dict['gapped']]

        # pad <EOS> & <SOS>
        Xs_f = [[_aa2id['<SOS>']] + list(seq)[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + list(seq)[::-1][:-1] for seq in Xs]

        # get sequence lengths
        Xs_len = [len(seq) for seq in Xs_f]
        lmax = max(Xs_len)

        # list to *.tensor
        Xs_f = [torch.tensor(seq) for seq in Xs_f]
        Xs_b = [torch.tensor(seq) for seq in Xs_b]

        # padding
        Xs_f = pad_sequence(Xs_f, batch_first=True)
        Xs_b = pad_sequence(Xs_b, batch_first=True)

        # embedding
        Xs_f = self.word_embeddings(Xs_f.type(dtype=torch.LongTensor))
        Xs_b = self.word_embeddings(Xs_b.type(dtype=torch.LongTensor))

        # packing the padded sequences
        Xs_f = pack_padded_sequence(Xs_f, Xs_len, batch_first=True, enforce_sorted=False)
        Xs_b = pack_padded_sequence(Xs_b, Xs_len, batch_first=True, enforce_sorted=False)

        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, batch_size, self.para_dict['hidden_dim']),
                          torch.zeros(1, batch_size, self.para_dict['hidden_dim']))
        ini_hc_state_b = (torch.zeros(1, batch_size, self.para_dict['hidden_dim']),
                          torch.zeros(1, batch_size, self.para_dict['hidden_dim']))

        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)

        # unpack outputs
        lstm_out_f, lstm_out_len = pad_packed_sequence(lstm_out_f, batch_first=True)
        lstm_out_b, _            = pad_packed_sequence(lstm_out_b, batch_first=True)

        lstm_out_valid_f = lstm_out_f.reshape(-1, self.para_dict['hidden_dim'])
        lstm_out_valid_b = lstm_out_b.reshape(-1, self.para_dict['hidden_dim'])

        idx_f = []
        [idx_f.extend([i*lmax+j for j in range(l)]) for i, l in enumerate(Xs_len)]
        idx_f = torch.tensor(idx_f)

        idx_b = []
        [idx_b.extend([i*lmax+j for j in range(l)][::-1]) for i, l in enumerate(Xs_len)]
        idx_b = torch.tensor(idx_b)

        lstm_out_valid_f = torch.index_select(lstm_out_valid_f, 0, idx_f)
        lstm_out_valid_b = torch.index_select(lstm_out_valid_b, 0, idx_b)

        lstm_out_valid = lstm_out_valid_f + lstm_out_valid_b

        # lstm hidden state to output space
        out = F.relu(self.fc1(lstm_out_valid))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        # compute scores
        scores = F.log_softmax(out, dim=1)

        return scores

    def objective(self):
        return nn.NLLLoss()

    def NLS_score(self, test_loader):

        scores = []
        self.eval()
        for n, (batch, batch_flatten) in enumerate(test_loader):
            actual_batch_size = len(batch)  # last iteration may contain less sequences
            seq_len = [len(seq) for seq in batch]
            seq_len_cumsum = np.cumsum(seq_len)
            out = self.forward(batch)
            out = np.vstack(out.detach().numpy())
            out = np.split(out, seq_len_cumsum)[:-1]
            batch_scores = []
            for i in range(actual_batch_size):
                pos_scores = []
                for j in range(seq_len[i]):
                    pos_scores.append(out[i][j, batch[i][j]])
                batch_scores.append(-sum(pos_scores) / seq_len[i])
            scores.append(batch_scores[0])
        return np.array(scores)

    def evaluate(self, outputs, labels):
         y_pred=[]
         for a in outputs:
             y_pred.append(np.argmax(a))
         y_true = np.array(labels).flatten()
         y_pred = np.array(y_pred)
         mat = confusion_matrix(y_true, y_pred)
         acc = accuracy_score(y_true, y_pred)
         mcc = matthews_corrcoef(y_true, y_pred)

         print('Test: ')
         print(mat)
         print('Accuracy = %.3f ,MCC = %.3f' % (acc, mcc))

         # return mat, acc, mcc

if __name__ == '__main__':

    para_dict = {'model_name': 'LSTM_Bi1',
                 'optim_name': 'Adam',
                 'step_size': 100,
                 'epoch': 200,
                 'batch_size' : 25000,
                 'learning_rate': 0.01,
                 'gapped':True,
                 'embedding_dim':64,
                 'hidden_dim':64,
                 'fixed_len':True}

    # train_data = OAS_data_loader.OAS_data_loader(
    #     index_file='./Models_generation/Benchmarks/OAS_dataset/data/OAS_meta_info.txt', output_field='Species',
    #     input_type='CDR3')
    # train_x = [x for x,y in train_data]
    # train_loader = torch.utils.data.DataLoader(train_x, batch_size=para_dict['batch_size'], drop_last=True,
    #                                            collate_fn=collate_fn)
    train_data = pd.read_csv('./Models_generation/Benchmarks/OAS_dataset/data/Human_train_seq.csv', sep='\t')
    train_x = OAS_data_loader.encode_index(data=train_data['seq'].values)
    train_loader = torch.utils.data.DataLoader(train_x, batch_size=para_dict['batch_size'], drop_last=False, collate_fn=collate_fn)
    para_dict['in_dim'] = len(aa2id_i[para_dict['gapped']])
    para_dict['out_dim'] =  len(aa2id_o[para_dict['gapped']])
    model = LSTM_Bi(para_dict)
    model.fit(train_loader)

    data = pd.read_csv('./Models_generation/Benchmarks/OAS_dataset/data/Human_train_seq.csv', sep='\t')
    train_x = OAS_data_loader.encode_index(data=data['seq'].values)
    train_mm = torch.utils.data.DataLoader(train_x, collate_fn=collate_fn)
    # human_mat, human_acc, human_mcc = model.evaluate(model.predict(test_loader),
    #                                                  np.vstack([i for _, i in test_loader]))
    output_human_train = model.NLS_score(train_mm)

    test_data = pd.read_csv('./Models_generation/Benchmarks/OAS_dataset/data/Human_test_seq.csv', sep='\t')
    test_x = OAS_data_loader.encode_index(data=test_data['seq'].values)
    test_loader = torch.utils.data.DataLoader(test_x, collate_fn=collate_fn)
    # human_mat, human_acc, human_mcc = model.evaluate(model.predict(test_loader),
    #                                                  np.vstack([i for _, i in test_loader]))
    output_human = model.NLS_score(test_loader)

    test_data = pd.read_csv('./Models_generation/Benchmarks/OAS_dataset/data/Rabbit_test_seq.csv',sep='\t')
    test_x = OAS_data_loader.encode_index(data=test_data['seq'].values)
    test_loader = torch.utils.data.DataLoader(test_x, collate_fn=collate_fn)
    # rabbit_mat, rabbit_acc, rabbit_mcc = model.evaluate(model.predict(test_loader),
    #                                                     np.vstack([i for _, i in test_loader]))
    output_rabbit = model.NLS_score(test_loader)

    test_data = pd.read_csv('./Models_generation/Benchmarks/OAS_dataset/data/Mouse_test_seq.csv', sep='\t')
    test_x = OAS_data_loader.encode_index(data=test_data['seq'].values)
    test_loader = torch.utils.data.DataLoader(test_x, collate_fn=collate_fn)
    # mouse_mat, mouse_acc, mouse_mcc = model.evaluate(model.predict(test_loader),
    #                                                     np.vstack([i for _, i in test_loader]))
    output_mouse = model.NLS_score(test_loader)

    test_data = pd.read_csv('./Models_generation/Benchmarks/OAS_dataset/data/Rhesus_test_seq.csv', sep='\t')
    test_x = OAS_data_loader.encode_index(data=test_data['seq'].values)
    test_loader = torch.utils.data.DataLoader(test_x, collate_fn=collate_fn)
    # rhesus_mat, rhesus_acc, rhesus_mcc = model.evaluate(model.predict(test_loader),
    #                                                     np.vstack([i for _, i in test_loader]))
    output_rhesus = model.NLS_score(test_loader)

    label = [-1 if a<1000 else 1 for a in range(2000)]
    output = np.concatenate((output_human,output_rabbit), axis=0)
    rabbit_fpr, rabbit_tpr, _ = roc_curve(np.array(label), np.array(output))
    AUC_score_rabbit = roc_auc_score(np.array(label), np.array(output))
    print('AUC score for rabbit ',AUC_score_rabbit)
    plt.plot(rabbit_fpr, rabbit_tpr, 'r', label='rabbit(AUC='+str(AUC_score_rabbit)+')')

    output = np.concatenate((output_human, output_mouse), axis=0)
    mouse_fpr, mouse_tpr, _ = roc_curve(np.array(label), np.array(output))
    AUC_score_mouse = roc_auc_score(np.array(label), np.array(output))
    print('AUC score for mouse ', AUC_score_mouse)
    plt.plot(mouse_fpr, mouse_tpr, 'g', label = 'mouse(AUC='+str(AUC_score_mouse)+')')

    output = np.concatenate((output_human, output_rhesus), axis=0)
    rhesus_fpr, rhesus_tpr, _ = roc_curve(np.array(label), np.array(output))
    AUC_score_rhesus = roc_auc_score(np.array(label), np.array(output))
    print('AUC score for rhesus ', AUC_score_rhesus)
    plt.plot(rhesus_fpr, rhesus_tpr, 'b', label = 'rhesus(AUC='+str(AUC_score_rhesus)+')')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.savefig(model.save_path)

    # plt.hist(output_human_train, histtype='step', normed=True, color='red', label='Human_train')
    # plt.hist(output_human, histtype='step', normed=True, color='orange', label='Human_test')
    # plt.hist(output_rabbit, histtype='step', normed=True, color='blue', label='Rabbit')
    # plt.hist(output_mouse, histtype='step', normed=True, color='green', label='Mouse')
    # plt.legend()
    # plt.show()
    # print()
    # print(para_dict)