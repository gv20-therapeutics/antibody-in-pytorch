from torch.utils.data import Dataset

aa_store = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa2id = dict(zip(aa_store,  list(range(len(aa_store)))))
id2aa = dict(zip(list(range(len(aa_store))), aa_store))

def one_of_k_encoding(x, allowable_set):
	if x not in allowable_set:
		raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
	return [x == s for s in allowable_set]

class ProteinSeqDataset(Dataset):
	def __init__(self, fn, fo):
		# initialize file path or list of file names
		with open(fn, 'r') as f:
            self.data = [l.strip('\n') for l in f]
		self.X = [[one_of_k_encoding(aa, aa_store) for aa in r] for r in self.data]

		with open(fo, 'r') as f:
			self.label = [l.strip('\n') for l in f]

		assert len(self.X) == len(self.label)

	def __getitem__(self,index):
		# 1. read one data from file
		# 2. preprocess the data
		# 3. return a data pair (X and y)
		return self.X[idx], self.label[idx]

	def __len__(self):
		# return the total size of the dataset
		return len(self.X)

#-------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMRNN_classifier(nn.Module):
    def __init__(self, hidden_layer_num, hidden_dim, dropout_rate, max_seq_len = 10):
        super(LSTMRNN_classifier, self).__init__()
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(20, hidden_dim, batch_first = False, 
                            num_layers = hidden_layer_num, dropout = dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, Xs):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(1,0,2)
        
        out, _ = self.lstm(X)
        out = out[-1,:,:].reshape(batch_size,-1) # use the last output as input for next layer
        out = torch.sigmoid(self.fc(out))

        return out

    def set_param(self, param_dict):
        pass

    def get_param(self):
        pass

