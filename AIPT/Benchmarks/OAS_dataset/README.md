## Benchmark dataset performance comparison 

### Description

Benchmark dataset consists of Observed Antibody Space (OAS) sequences. We used CDR1, CDR2 and CDR3 sequences to analyze the performance for classification of Naive B cells and Memory B cells. Accuracy score and Matthew's Correlation Coefficient (MCC) was used to compare the results. The following table shows the performance for CNN, LSTM-RNN and Bi-LSTM models on the benchmark dataset:

### Tables of comparison

#### CDR3 sequences:

| Models | Accuracy |  MCC  |
| :----: | :------: | :----: |
| CNN | 0.764 | 0.530 |
| LSTM-RNN | 0.734 | 0.44 |
| Bi-LSTM | 0.74 | 0.452 |

#### CDR1, CDR2 & CDR3 sequences combined:

| Models | Accuracy |  MCC  |
| :----: | :------: | :----: |
| CNN | 0.796 | 0.599 |
| LSTM-RNN | 0.763 | 0.499 |
| Bi-LSTM | 0.697 | 0.402 |
