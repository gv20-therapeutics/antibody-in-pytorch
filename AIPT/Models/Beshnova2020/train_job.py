
def main():
    import os
    import argparse
    import warnings

    warnings.filterwarnings("ignore")
    import pandas as pd
    from AIPT.Models.Beshnova2020.CNN import CNN
    from AIPT.Utils.plotting import roc_from_models
    from AIPT.Models.Beshnova2020.data import load_data, get_train_test_loaders

    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_len', type=int, default=11)
    parser.add_argument('--embedding_dim', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=10**-3)
    parser.add_argument('--epoch', type=int, default=10**-3)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--run_name', type=str, default='default_run_name')
    parser.add_argument('--run_dir', type=str, default='default_run_dir')
    parser.add_argument('--work_path', type=str, default='/opt/ml/model')
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_DATA'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    parser.add_argument('--index_file', type=str, default='OAS_index.txt')

    # architecture spec
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--conv1_n_filters', type=int, default=8)
    parser.add_argument('--conv2_n_filters', type=int, default=16)
    parser.add_argument('--conv1_filter_dim1', type=int, default=2)
    parser.add_argument('--conv2_filter_dim1', type=int, default=2)
    parser.add_argument('--max_pool_filter_dim1', type=int, default=2)
    parser.add_argument('--fc_hidden_dim', type=int, default=10)
    # parser.add_argument('--stride', type=int, default=1)
    # parser.add_argument('--padding', type=int, default=0)


    para_dict = vars(parser.parse_args())

    # aipt_path = '/home/ec2-user/SageMaker/antibody-in-pytorch/'
    aipt_path = './'


    seq_dir = para_dict["data_dir"]
    model_dir = 'AIPT/Models/Beshnova2020'
    index_fn = para_dict['index_file']
    index_path = os.path.join(aipt_path, model_dir, index_fn)
    cell_types = [
        "Naive-B-Cells",
        "Memory-B-Cells",
    ]
    para_dict['classes'] = cell_types
    index_df = pd.read_csv(index_path, sep="\t")
    data = load_data(index_df, seq_dir, cell_types, seq_len=para_dict['seq_len'])
    train_loader, test_loader = get_train_test_loaders(data)

    ## models

    run_dir = para_dict['run_dir']
    pca_para_dict = para_dict.copy()
    pca_para_dict['model_name'] = 'pca'
    pca_para_dict['model_path'] = os.path.join(para_dict['work_path'], pca_para_dict['model_name'])
    pca_dir = os.path.join(run_dir, 'pca')
    pca_para_dict['log_dir'] = os.path.join(pca_dir, 'logs')
    pca_model = CNN.pca_model(pca_para_dict)

    general_para_dict = para_dict.copy()
    general_para_dict['model_name'] = 'general'
    general_para_dict['model_path'] = os.path.join(para_dict['work_path'], general_para_dict['model_name'])
    general_dir = os.path.join(run_dir, 'general')
    general_para_dict['log_dir'] = os.path.join(general_dir, 'logs')
    general_para_dict['log_dir'] = os.path.join(general_para_dict['log_dir'], 'general')
    general_model = CNN.general_model(general_para_dict)

    ## Train

    pca_model.fit(train_loader, test_loader=test_loader)
    general_model.fit(train_loader, test_loader=test_loader)

    # Evaluate

    def model_evaluation(data_loader, data_name, figure_dir, figure_basename, title_basename):
        figure_path = os.path.join(figure_dir, f'{figure_basename}_{data_name}.png')
        roc_from_models(
            {
                pca_model.name.upper(): pca_model,
                general_model.name.capitalize(): general_model,
            },
            {
                pca_model.name.upper(): data_loader,
                general_model.name.capitalize(): data_loader,
            },
            title=f"{title_basename} ({data_name.capitalize()})",
            save_path=figure_path,
            show=False
        )

    figure_dir = os.path.join(para_dict['output_data_dir'], 'figures')
    os.makedirs(figure_dir)
    figure_basename = 'memory_naive_roc'
    title_basename = 'Memory vs. Naive B-Cell Classification'

    model_evaluation(train_loader, 'train', figure_dir, figure_basename, title_basename)
    print('\n')
    model_evaluation(test_loader, 'test', figure_dir, figure_basename, title_basename)

