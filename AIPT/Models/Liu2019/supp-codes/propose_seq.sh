mkdir ../results
mkdir ../results/seq_gen
mkdir log
bash utils/sub_ensemble.sh Easy_classification_0615_holdouttop_reg reg
bash utils/sub_ensemble.sh Easy_classification_0622_reg reg
bash utils/sub_ensemble.sh Easy_classification_0604_holdouttop_reg reg
bash utils/sub_ensemble.sh Easy_classification_0604_holdouttop class
