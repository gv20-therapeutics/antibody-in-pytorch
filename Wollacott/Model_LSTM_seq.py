from torch.utils.data import Dataset
from ..Utils.model import Model
from ..Utils import loader
import numpy as np

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


# class ProteinSeqDataset(Dataset):   # plain text file containing amino acid sequences
#     def __init__(self, fn, gapped=True):
#         # load data
#         with open(fn, 'r') as f:
#             self.data = [l.strip('\n') for l in f]
#
#         # char to id
#         self.data = [[aa2id_i[gapped][c] for c in r] for r in self.data]
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]
    
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
        batch_size = self.para_dict['batch_size']
        # print(_aa2id)
        _aa2id =  aa2id_i[para_dict['gapped']]

        # print(Xs)
        # pad <EOS> & <SOS>
        Xs_f = [[_aa2id['<SOS>']] + list(seq)[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + list(seq)[::-1][:-1] for seq in Xs]
        
        # get sequence lengths
        X_len = len(Xs_f[0])

        # print(Xs_f)
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
        Xs_f = [[_aa2id['<SOS>']] + seq[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + seq[::-1][:-1] for seq in Xs]
        
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
        Xs_f = self.word_embeddings(Xs_f)
        Xs_b = self.word_embeddings(Xs_b)
        
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
        print(scores)
        
        return scores
    
    def objective(self):
        return nn.NLLLoss()

    # def set_param(self, self.para_dict):
    #     try:
    #         for pn, _ in self.named_parameters():
    #             exec('self.%s.data = torch.tensor(self.para_dict[pn])' % pn)
    #         self.hidden_dim = self.para_dict['hidden_dim']
    #         self.fixed_len = self.para_dict['fixed_len']
    #         self.forward = self.forward_flen if self.fixed_len else self.forward_vlen
    #     except:
    #         print('Unmatched parameter names or shapes.')

    # def get_param(self):
    #     self.para_dict = {}
    #     for pn, pv in self.named_parameters():
    #         self.para_dict[pn] = pv.data.cpu().numpy()
    #     self.para_dict['hidden_dim'] = self.hidden_dim
    #     self.para_dict['fixed_len'] = self.fixed_len
    #     return self.para_dict

#------------------------------------------------------------------------------------------
from tqdm import tqdm
import numpy as np
import sys

# class ModelLSTM:
#     def __init__(self, embedding_dim=64, hidden_dim=64, device='cpu', gapped=True, fixed_len=True):
#         self.gapped = gapped
#         in_dim, out_dim = len(aa2id_i[gapped]), len(aa2id_o[gapped])
#         self.nn = LSTM_Bi(in_dim, embedding_dim, hidden_dim, out_dim, device, fixed_len)
#         self.to(device)

#     def fit(self, trn_fn, vld_fn, n_epoch=10, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp=None):
#         # loss function and optimization algorithm
#         loss_fn = self.objective()
#         op = torch.optim.Adam(self.nn.parameters(), lr=lr)

#         # to track minimum validation loss
#         min_loss = np.inf

        # dataset and dataset loader
        # trn_data = ProteinSeqDataset(trn_fn, self.gapped)
        # vld_data = ProteinSeqDataset(vld_fn, self.gapped)
        # if trn_batch_size == -1: trn_batch_size = len(trn_data)
        # if vld_batch_size == -1: vld_batch_size = len(vld_data)
        # trn_dataloader = torch.utils.data.DataLoader(trn_data, trn_batch_size, True, collate_fn=collate_fn)
        # vld_dataloader = torch.utils.data.DataLoader(vld_data, vld_batch_size, False, collate_fn=collate_fn)

#         for epoch in range(n_epoch):
#             # training
#             self.nn.train()
#             loss_avg, acc_avg, cnt = 0, 0, 0
#             with tqdm(total=len(trn_data), desc='Epoch {:03d} (TRN)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
#                 for batch, batch_flatten in trn_dataloader:
#                     # targets
#                     batch_flatten = torch.tensor(batch_flatten, device=self.nn.device)

#                     # forward and backward routine
#                     self.nn.zero_grad()
#                     scores = self.nn(batch, aa2id_i[self.gapped])
#                     loss = loss_fn(scores, batch_flatten)
#                     loss.backward()
#                     op.step()

#                     # compute statistics
#                     L = len(batch_flatten)
#                     predicted = torch.argmax(scores, 1)
#                     loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
#                     corr = (predicted == batch_flatten).data.cpu().numpy()
#                     acc_avg = (acc_avg * cnt + sum(corr)) / (cnt + L)
#                     cnt += L

#                     # update progress bar
#                     pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg), 'acc':  '{:.6f}'.format(acc_avg)})
#                     pbar.update(len(batch))

#             # validation
#             self.nn.eval()
#             loss_avg, acc_avg, cnt = 0, 0, 0
#             with torch.set_grad_enabled(False):
#                 with tqdm(total=len(vld_data), desc='          (VLD)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
#                     for batch, batch_flatten in vld_dataloader:
#                         # targets
#                         batch_flatten = torch.tensor(batch_flatten)

#                         # forward routine
#                         scores = self.nn(batch, aa2id_i[self.gapped])
#                         loss = loss_fn(scores, batch_flatten)

#                         # compute statistics
#                         L = len(batch_flatten)
#                         predicted = torch.argmax(scores, 1)
#                         loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
#                         corr = (predicted == batch_flatten).data.cpu().numpy()
#                         acc_avg = (acc_avg * cnt + sum(corr)) / (cnt + L)
#                         cnt += L

#                         # update progress bar
#                         pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg), 'acc':  '{:.6f}'.format(acc_avg)})
#                         pbar.update(len(batch))

#             # save model
#             if loss_avg < min_loss and save_fp:
#                 min_loss = loss_avg
#                 self.save('{}/lstm_{:.6f}.npy'.format(save_fp, loss_avg))

#     def eval(self, fn, batch_size=512):
#         # dataset and dataset loader
#         data = ProteinSeqDataset(fn, self.gapped)
#         if batch_size == -1: batch_size = len(data)
#         dataloader = torch.utils.data.DataLoader(data, batch_size, True, collate_fn=collate_fn)

#         self.nn.eval()
#         scores = np.zeros(len(data), dtype=np.float32)
#         sys.stdout.flush()
#         with torch.set_grad_enabled(False):
#             with tqdm(total=len(data), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
#                 for n, (batch, batch_flatten) in enumerate(dataloader):
#                     actual_batch_size = len(batch)  # last iteration may contain less sequences
#                     seq_len = [len(seq) for seq in batch]
#                     seq_len_cumsum = np.cumsum(seq_len)
#                     out = self.nn(batch, aa2id_i[self.gapped]).data.cpu().numpy()
#                     out = np.split(out, seq_len_cumsum)[:-1]
#                     batch_scores = []
#                     for i in range(actual_batch_size):
#                         pos_scores = []
#                         for j in range(seq_len[i]):
#                             pos_scores.append(out[i][j, batch[i][j]])
#                         batch_scores.append(-sum(pos_scores) / seq_len[i])
#                     scores[n*batch_size:(n+1)*batch_size] = batch_scores
#                     pbar.update(len(batch))
#         return scores

#     def save(self, fn):
#         self.para_dict = self.nn.get_param()
#         self.para_dict['gapped'] = self.gapped
#         np.save(fn, self.para_dict)

#     def load(self, fn):
#         self.para_dict = np.load(fn, allow_pickle=True).item()
#         self.gapped = self.para_dict['gapped']
#         self.nn.set_param(self.para_dict)

#     def to(self, device):
#         self.nn.to(device)
#         self.nn.device = device

#     def summary(self):
#         for n, w in self.nn.named_parameters():
#             print('{}:\t{}'.format(n, w.shape))
# #        print('LSTM: \t{}'.format(self.nn.lstm_f.all_weights))
#         print('Fixed Length:\t{}'.format(self.nn.fixed_len) )
#         print('Gapped:\t{}'.format(self.gapped))
#         print('Device:\t{}'.format(self.nn.device))
            
if __name__ == '__main__':

    para_dict = {'num_samples': 1000,
                 'seq_len': 10,
                 'model_name': 'LSTM_Bi',
                 'optim_name': 'Adam',
                 'step_size': 10,
                 'epoch': 20,
                 'batch_size' : 20,
                 'learning_rate': 0.01,
                 'gapped':True,
                 'embedding_dim':64,
                 'hidden_dim':64,
                 'fixed_len':True}

    train_loader, test_loader = loader.synthetic_data_loader(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'], aa_list='ACDEFGHIKLMNPQRSTVWY_', test_size=0.3, batch_size=para_dict['batch_size'])
    para_dict['in_dim'] = len(aa2id_i[para_dict['gapped']])
    para_dict['out_dim'] =  len(aa2id_o[para_dict['gapped']])
    model = LSTM_Bi(para_dict)
    model.fit(train_loader)
    # print(test_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])

    # len(labels)
    mat, acc, mcc = model.evaluate(output, labels)
