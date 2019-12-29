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

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_classifier(nn.Module):
    def __init__(self, n_filter, filter_size, fc_hidden_dim, dropout_rate, device, max_seq_len = 10):
        super(CNN_classifier, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv1d(in_channels = 20, out_channels = n_filter, kernel_size=filter_size, 
                                stride = 1, padding = 0)
        self.pool = nn.MaxPool1d(kernel_size = 2, stride = 1)
        self.fc1 = nn.Linear(in_features = (max_seq_len-filter_size) * n_filter, out_features = fc_hidden_dim)
        self.fc2 = nn.Linear(in_features = fc_hidden_dim, out_features = 1)

    def forward(self, Xs):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)
        
        out = F.dropout(self.conv1(X), p = self.dropout_rate)
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))

        return out

    def set_param(self, param_dict):
        pass

    def get_param(self):
        pass



