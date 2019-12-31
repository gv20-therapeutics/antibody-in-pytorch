
| Model name | Input | Output | Loss Function | Model structure |
| CNN classifier | Amino acid sequence of length 10 (CDRH3 only) without gap | Probability of binding | Binary cross entropy | LSTM-RNN (3 layers, 40 hidden nodes, dropout rate = 0.1) + sigmoid activation |  
| LSTM RNN classifier | Amino acid sequence of length 10 (CDRH3 only) without gap | Probability of binding | Binary cross entropy | CNN (400 filters, kernel size 3, stride 1) + max pooling + FC + Relu activateion | 

