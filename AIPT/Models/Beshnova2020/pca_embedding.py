import pandas as pd
import numpy as np
import torch
import AIPT.Utils.Dev.dev_utils as dev_utils

DEBUG_MODE = True
if DEBUG_MODE:
    dev_utils.get_mod_time(__file__, verbose=True)

AA_PCA_FEATURES = pd.read_csv('AAidx_PCA.txt', sep='\t').sort_index().to_numpy()

def embed_seq(aa_seq):
    encodings = []
    index = AA_PCA_FEATURES.loc if type(aa_seq) == str else AA_PCA_FEATURES.iloc
    for aa in aa_seq:
        encodings.append(index[aa.item()])
    encodings =  np.vstack(encodings)
    return torch.Tensor(encodings)

# def embed_batch(aa_seq_batch):
#     encoded_batch = map(embed_seq, aa_seq_batch)
#     batch =  torch.stack(tuple(encoded_batch))
#     return batch

def embedding_fn(num_embeddings, embedding_dim):
    torch_embedding_fn = torch.nn.Embedding(num_embeddings, embedding_dim)
    torch_embedding_fn.weight.data.copy_(torch.from_numpy(AA_PCA_FEATURES))
    torch_embedding_fn.requires_grad = False
    return torch_embedding_fn
