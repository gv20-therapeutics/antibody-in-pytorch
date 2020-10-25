import argparse
import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
import AIPT.Models.Beshnova2020.CNN
import AIPT.Utils.logging
import AIPT.Utils.plotting
import AIPT.Utils.Dev.dev_utils as dev_utils

def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_len', type=int, default=11)
    parser.add_argument('--embedding_dim', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=10**-3)
    parser.add_argument('--run_name', type=str, default='default_run_name')
    parser.add_argument('--run_dir', type=str, default='default_run_dir')
    parser.add_argument('--work_path', type=str, default='/opt/ml/model')
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_DATA'])
    para_dict = parser.parse_args()

    # aipt_path = '/home/ec2-user/SageMaker/antibody-in-pytorch/'
    aipt_path = './'
    aipt_reload = dev_utils.get_aipt_reload_fn(aipt_path)

    '''
    set up paths
    '''

    # aipt_dir = '/home/ec2-user/SageMaker/antibody-in-pytorch/AIPT'  # replace with your own aipt path
    aipt_dir = aipt_path

    # seq_dir = os.path.join(aipt_dir, "Benchmarks/OAS_dataset/data/seq_db")
    seq_dir = para_dict.data_dir
    model_dir = 'AIPT/Models/Beshnova2020'
    model_dir_abs = os.path.join(aipt_path, model_dir)
    index_fn = "OAS_index.txt"
    index_path = os.path.join(aipt_path, model_dir, index_fn)
    cell_types = [
        "Naive-B-Cells",
        "Memory-B-Cells",
    ]  # todo: this is confusing - doesn't refer to "Species"
    para_dict['classes'] = cell_types

    index_df = pd.read_csv(index_path, sep="\t")

    file_names = index_df['file_name']

    data_dfs = []
    for index, row in index_df.iterrows():
        file_name = row['file_name']
        df = pd.read_csv(os.path.join(seq_dir, f'{file_name}.txt'), sep='\t')
        length_df = df.apply(lambda row: len(row['CDR3_aa']), axis=1)
        data_df = df[length_df == 12]
        data_df['BType'] = row['BType']
        data_df = data_df[['CDR3_aa', 'BType']]
        data_dfs.append(data_df)

    data = pd.concat(data_dfs)

    from AIPT.Benchmarks.OAS_dataset import OAS_data_loader
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, WeightedRandomSampler

    np.random.seed(0)
    torch.manual_seed(0)

    def get_balanced_data_loader(data, batch_size=32):
        # useful example: https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/20
        # Compute samples weight (each sample should get its own weight)
        label = torch.Tensor(data['label'].values).type(torch.int8)
        class_sample_count = torch.tensor(
            [(label == t).sum() for t in torch.unique(label, sorted=True)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in label])

        # Create sampler, dataset, loader
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        seq_encodings = OAS_data_loader.encode_index(data=data['CDR3_aa'])
        #     dataset = TensorDataset(torch.Tensor(seq_encodings), label)
        btypes = data['label'].values
        dataset = list(zip(seq_encodings, btypes))
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        return loader

    def get_data_loader(data, batch_size=32):
        seq_encodings = OAS_data_loader.encode_index(data=data['CDR3_aa'])
        btypes = data['label'].values
        loader = DataLoader(list(zip(seq_encodings, btypes)), shuffle=True, batch_size=batch_size, drop_last=True)
        return loader

    data['label'] = data.apply(lambda row: cell_types.index(row['BType']), axis=1)
    train_data, test_data = train_test_split(data, train_size=0.8)
    train_loader = get_balanced_data_loader(train_data)
    # train_loader = get_data_loader(train_data)
    test_loader = get_data_loader(test_data)


    # aipt_reload(AIPT.Models.Beshnova2020.CNN)
    # aipt_reload(AIPT.Utils.logging)
    # aipt_reload(AIPT.Utils.plotting)
    from AIPT.Models.Beshnova2020.CNN import CNN
    import AIPT.Models.Beshnova2020.pca_embedding as pca_embedding
    from AIPT.Utils.logging import today, current_time
    from AIPT.Utils.plotting import plot_roc_curves
    import os


    # print('LOG DIR:', para_dict['log_dir'])

    ## Tensorboard

    import subprocess as sp

    start_tensorboard = False

    if start_tensorboard:
        reload_interval = "15"  # seconds
        tensorboard_proc = sp.Popen(
            [
                "tensorboard",
                "--logdir",
                para_dict["log_dir"],
            ],
            universal_newlines=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
        )

    ## models
    run_dir = para_dict['run_dir']
    pca_para_dict = para_dict.copy()
    pca_para_dict['model_name'] = 'pca'
    pca_dir = os.path.join(run_dir, 'pca')
    pca_para_dict['log_dir'] = os.path.join(pca_dir, 'logs')
    pca_embedding_fn = pca_embedding.embedding_fn(20, pca_para_dict['embedding_dim'])
    pca_model = CNN(pca_para_dict, pca_embedding_fn)

    general_para_dict = para_dict.copy()
    general_para_dict['model_name'] = 'general'
    general_dir = os.path.join(run_dir, 'general')
    pca_para_dict['log_dir'] = os.path.join(general_dir, 'logs')
    general_para_dict['log_dir'] = os.path.join(general_para_dict['log_dir'], 'general')
    general_embedding_fn = nn.Embedding(20, general_para_dict['embedding_dim'])
    general_model = CNN(general_para_dict, general_embedding_fn)

    ## Train

    pca_model.fit(train_loader, test_loader=test_loader)
    general_model.fit(train_loader, test_loader=test_loader)

    figure_dir = os.path.join(para_dict['log_dir'],'figures')
    figure_path = os.path.join(figure_dir, 'memory_naive_roc_train.png')

    pca_output, pca_labels, pca_loss = pca_model.predict(train_loader)
    pca_model.evaluate(pca_output, pca_labels)
    general_output, general_labels, general_loss = general_model.predict(train_loader)
    general_model.evaluate(general_output, general_labels)
    plot_roc_curves(
        [pca_output[:, 1], general_output[:, 1]],
        [pca_labels, general_labels],
        ["PCA Embedding", "General Embedding"],
        title="Memory vs. Naive B-cell Classification (Train)",
        save_path=figure_path
    )

    figure_path = os.path.join(figure_dir, 'memory_naive_roc_test.png')

    general_output, general_labels, general_loss = general_model.predict(test_loader)
    general_model.evaluate(general_output, general_labels)
    pca_output, pca_labels, pca_loss = pca_model.predict(test_loader)
    pca_model.evaluate(pca_output, pca_labels)
    plot_roc_curves(
        [pca_output[:, 1], general_output[:, 1]],
        [pca_labels, general_labels],
        ["PCA Embedding", "General Embedding"],
        title="Memory vs. Naive B-cell Classification (Test)",
        save_path=figure_path
    )
