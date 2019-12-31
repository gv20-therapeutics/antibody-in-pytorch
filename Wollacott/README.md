| Model name | Input | Output | Loss Function | Model structure |
| ---------- | ----- | ------ | ------------- | --------------- |
| LSTM seq | Amino acid sequence of length ~ 150 (gapped or without gap) | predicted full length sequence | NLLLoss (negative log likelihood, following log-softmax equal CrossEntropyLoss | Word embedding + Bi-directional LSTM + 3 * (FC + Relu) + log softmax | 
