# Machine learning models for antibody sequences in PyTorch

## Introduction

Recently, more people are realizing the use of machine learning, especially deep learning, in helping to understand antibody sequences in terms of binding specificity, therapeutic potential, and developability. Several models have been proposed and shown excellent performance in different datasets. We believe there should be an optional solution of modeling antibody sequences, because if otherwise, people can use transfer learning to keep the good "knowledge" and train a minimal amount of parameters for specific tasks. Therefore, we create this public repo to collect and re-implement (if needed) public available machine learning models in PyTorch.

## Requirements

* Pytorch
* Pandas
* Numpy
* Scikit-learn

## Clone the repository to local machine

```bash
git clone https://github.com/gv20-therapeutics/antibody-in-pytorch.git      # Clone antibody-in-pytorch source code
cd gv20-therapeutics
```

## How to run?

**Command to run:** python -m antibody-in-pytorch.(Name of the folder).(Name of the model)

```bash 
python antibody-in-pytorch.Mason.Model_CNN_classifier_v1  # Example to run Mason's CNN model
```

## References

1. Mason et al., Deep learning enables therapeutic antibody optimization in mammalian cells by deciphering high-dimensional protein sequence space. [https://www.biorxiv.org/content/10.1101/617860v3]
2. Wollacott et al., Quantifying the nativeness of antibody sequences using long short-term memory networks. [https://academic.oup.com/peds/advance-article/doi/10.1093/protein/gzz031/5554642]

## Related works

1. Davidsen et al., Deep generative models for T cell receptor protein sequences. [https://elifesciences.org/articles/46935]
2. Hu et al., ACME: pan-specific peptide–MHC class I binding prediction through attention-based deep neural networks. [https://academic.oup.com/bioinformatics/article-abstract/35/23/4946/5497763]

## Contributors (alphabetical order)

* [Qingyang Ding](https://github.com/qid12)
* [Suraj Gattani](https://github.com/suraj-gattani)
* [Xihao Hu](https://github.com/huxihao)
