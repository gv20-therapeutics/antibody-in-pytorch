# Antibody CDR design pipeline

> Code for running ensemble gradient ascent optimization on panning derived sequences and reproducing figures in the paper.

## Running Ens_Grad with pretrained model
The `single_net.py` in `utils` runs gradient ascent based optimization on user specified seed sequences using a single pretrained neural network and user specified hyperparameters. Seed sequences are located in `utils/seeds` in h5py format which stores the padded and one hot encoded seed sequences. Pretrained models are located in `data/` where folders starting with *Easy_classification* stand for different training datasets.

```python single_opt.py <rootpath> <network> <resultdir> <k> <stepsize> <seed dir> <task type>```

The `sub_ensemble.sh` launches a set of jobs which runs gradient ascent based optimization using multiple hyperparameter setups and ensemble of models pretrained on a specific dataset using different architectures. All generated sequences are located under `results/plots/seq_gen/<dataset name>`.

```bash sub_ensemble.sh <dataset name> <task type>```

We prepared a single script `propose_seq.sh` which reproduces ensemble gradient ascent optimization using all pretrained networks, and `postprocess.py` which runs voting and thresholding using ensemble confidence lower bond as described in the paper, and compares with sequences proposed by other methods and seed sequences. Run `plot2.py`
 to reproduce all the plots in the paper. All generated plots are located under `results/plots`.