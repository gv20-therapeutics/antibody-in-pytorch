# Machine learning models for antibody sequences in PyTorch

## Introduction

Recently, more people are realizing the use of machine learning, especially deep learning, in helping to understand antibody sequences in terms of binding specificity, therapeutic potential, and developability. Several models have been proposed and shown excellent performance in different datasets. We believe there should be an optional solution of modeling antibody sequences, because if otherwise, people can use transfer learning to keep the good "knowledge" and train a minimal amount of parameters for specific tasks. Therefore, we create this public repo to collect and re-implement (if needed) public available machine learning models in PyTorch.

## Requirements

* Pytorch
* Pandas
* Numpy
* Scikit-learn

## Download and install the package 

```bash
git clone https://github.com/gv20-therapeutics/antibody-in-pytorch.git
cd antibody-in-pytorch.git
python setup.py install
```

## Ways to run the machine learning models

Directly run from the command line:
```bash 
AIPT --help
```
Run from a local python script
```bash 
python runner.py --help
```
Run as a module
```bash 
python -m AIPT --help
```
Run from the entry point in the module
```bash 
python -m AIPT.entry_point --help
```
Run without any parameter will go through all the test() functions.

## Antibody datasets
* SAAB and OAS [http://antibodymap.org/]
* SAbDab [http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/]
* cAb-Rep [https://cab-rep.c2b2.columbia.edu/tools/]

## References

1. Hu and Liu 2019, DeepBCR: Deep learning framework for cancer-type classification and binding affinity estimation using B cell receptor repertoires. [https://www.biorxiv.org/content/10.1101/731158v1]
2. Mason et al. 2020, Deep learning enables therapeutic antibody optimization in mammalian cells by deciphering high-dimensional protein sequence space. [https://www.biorxiv.org/content/10.1101/617860v3]
3. Liu et al. 2019, Antibody complementarity determining region design using high-capacity machine learning. [https://doi.org/10.1093/bioinformatics/btz895]
4. Wollacott et al. 2019, Quantifying the nativeness of antibody sequences using long short-term memory networks. [https://academic.oup.com/peds/advance-article/doi/10.1093/protein/gzz031/5554642]
5. Chen et al. 2020, Predicting Antibody Developability from Sequence using Machine Learning. [https://www.biorxiv.org/content/10.1101/2020.06.18.159798v1]
6. Beshnova et al. 2020, De novo prediction of cancer-associated T cell receptors for noninvasive cancer detection. [https://stm.sciencemag.org/content/12/557/eaaz3738]

## Related works

1. Davidsen et al., Deep generative models for T cell receptor protein sequences. [https://elifesciences.org/articles/46935]
2. Hu et al., ACME: pan-specific peptide–MHC class I binding prediction through attention-based deep neural networks. [https://academic.oup.com/bioinformatics/article-abstract/35/23/4946/5497763]

## Contributors (alphabetical order)

* [Qingyang Ding](https://github.com/qid12)
* [Suraj Gattani](https://github.com/suraj-gattani)
* [Xihao Hu](https://github.com/huxihao)
* [Yaowen Chen](https://github.com/achenge07)
