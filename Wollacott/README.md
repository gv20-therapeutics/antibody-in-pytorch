Wollacott et al. developed a bi-directional long short-term memory (LSTM) network model, and use the model to quantify the nativeness of antibody sequences. 
The model was trained using large amount of antibody sequences, and scores sequences for their similarity to naturally occuring antibodies.

![avatar](LSTMseq_model_struct.png)

| Model name | Input | Output | Loss Function | Model structure |
| ---------- | ----- | ------ | ------------- | --------------- |
| LSTM seq | Amino acid sequence of length ~ 150 (gapped or without gap) | predicted full length sequence | NLLLoss (negative log likelihood, following log-softmax equal CrossEntropyLoss | Word embedding + Bi-directional LSTM + 3 * (FC + Relu) + log softmax | 

**Reference**

Wollacott et al., Quantifying the nativeness of antibody sequences using long short-term memory networks. [https://academic.oup.com/peds/advance-article/doi/10.1093/protein/gzz031/5554642]
