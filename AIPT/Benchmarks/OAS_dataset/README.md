## Benchmark dataset performance comparison 

### Description

Benchmark dataset consists of Observed Antibody Space (OAS) sequences. We used CDR1, CDR2 and CDR3 sequences to analyze the performance for classification of Naive B cells and Memory B cells. Accuracy score and Matthew's Correlation Coefficient (MCC) metrics were used to compare the results. The following table shows the performance for CNN, LSTM-RNN and Bi-LSTM models on the benchmark dataset:

### Hyperparameter optimization 

#### Mason2020 CNN classifier:

1. n-filter: IntegerParameter(300, 600),
2. filter-size: IntegerParameter(2, 8),
3. fc-hidden-dim: IntegerParameter(50, 400),
4. epoch: IntegerParameter(25, 50),
5. learning-rate: ContinuousParameter(1e-4,1e-1)

#### Mason2020 LSTM-RNN classifier:

1. hidden-layer-num: IntegerParameter(2, 10),
2. hidden-dim: IntegerParameter(50, 300),
3. epoch: IntegerParameter(25, 75),
4. learning-rate: ContinuousParameter(1e-4,1e-1)

#### Wollacott2019 Bi-LSTM classifier:

1. embedding-dim: IntegerParameter(50, 300),
2. hidden-dim: IntegerParameter(50, 300),
3. epoch: IntegerParameter(25, 75),
4. learning-rate: ContinuousParameter(1e-4,1e-1)
                       
#### Train and test sequence count:

- Number of train sequences: 164,687
- Number of test sequences: 95,551

### Tables of comparison

#### CDR3 sequences:

| Models | Accuracy |  MCC  |  Best parameters after optimization |
| :----: | :------: | :----: | :--: | 
| CNN | 0.730 | 0.476 | epoch=68, n_filter=389, fc_hidden_dim=157, filter_size=4, learning_rate=0.002 |
| LSTM-RNN | 0.715 | 0.444 | epoch=60, hidden_dim=175, hidden_layer_num=3, learning_rate=0.002 |
| Bi-LSTM | 0.647 | 0.303 | epoch=28, embedding_dim=251, hidden_dim=135, learning_rate=0.001 |

#### CDR1, CDR2 & CDR3 sequences combined:

| Models | Accuracy |  MCC  | Best parameters after optimization |
| :----: | :------: | :----: | :--: | 
| CNN | 0.796 | 0.599 | epoch=68, n_filter=389, fc_hidden_dim=157, filter_size=4, learning_rate=0.002 |
| LSTM-RNN | 0.759 | 0.518 | epoch=60, hidden_dim=175, hidden_layer_num=3, learning_rate=0.002 |
| Bi-LSTM | 0.697 | 0.398 | epoch=28, embedding_dim=251, hidden_dim=135, learning_rate=0.001 |
