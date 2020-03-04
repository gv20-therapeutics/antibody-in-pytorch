import numpy as np
import optparse

__version__ = '0.0.1'

def main():
    print("""Welcome to use Antibody-In-PyTorch (AIPT) 
    version {}
    """.format(__version__))

    parser = optparse.OptionParser()

    parser.add_option('--num-samples', type=int, default=1000)
    parser.add_option('--seq-len', type=int, default=20)
    parser.add_option('--batch-size', type=int, default=5000)
    parser.add_option('--dataset', type=str, default='OAS')
    parser.add_option('--model-name', type=str, default='Wollacott2019_Bi_LSTM')
    parser.add_option('--optim-name', type=str, default='Adam')
    parser.add_option('--epoch', type=int, default=5)
    parser.add_option('--learning-rate', type=float, default=1e-3)
    parser.add_option('--step-size', type=int, default=10)
    parser.add_option('--dropout-rate', type=float, default=0.5)
    parser.add_option('--random-state', type=int, default=100)  # For splitting the data

    parser.add_option('--n_filter', type=int, default=400)  # CNN model
    parser.add_option('--filter-size', type=int, default=3)  # CNN model
    parser.add_option('--fc-hidden-dim', type=int, default=50)  # CNN model
    parser.add_option('--hidden-layer-num', type=int, default=3)  # LSTM-RNN model
    parser.add_option('--hidden-dim', type=int, default=64)  # Bi_LSTM model
    parser.add_option('--embedding-dim', type=int, default=64)  # Bi_LSTM model


    args, _ = parser.parse_args()

    para_dict = {'batch_size': args.batch_size,
                 'optim_name': args.optim_name,
                 'epoch': args.epoch,
                 'learning_rate': args.learning_rate,
                 'step_size': args.step_size,
                 'n_filter': args.n_filter,
                 'filter_size': args.filter_size,
                 'fc_hidden_dim': args.fc_hidden_dim,
                 'dropout_rate': args.dropout_rate,
                 'hidden_layer_num': args.hidden_layer_num,
                 'hidden_dim': args.hidden_dim,
                 'embedding_dim': args.embedding_dim,
                 'species_type': ['human', 'mouse', 'rabbit', 'rhesus'],
                 'fixed_len': False,
                 'gapped': True,
                 'pad': False}

    if args.dataset == 'Test':
        """
        Runs all models or specific model on synthetic data
        """
        if args.model_name == 'Mason2020_CNN' or args.model_name == 'All':
            from .Models.Mason2020.CNN import test
            test()
        if args.model_name == 'Mason2020_LSTM' or args.model_name == 'All':
            from .Models.Mason2020.LSTM_RNN import test
            test()
        if args.model_name == 'Wollacott2019' or args.model_name == 'All':
            from .Models.Wollacott2019.Bi_LSTM import test
            test()

    elif args.dataset == 'OAS':
        """
        Loads the OAS dataset and creates the train & test loader
        Runs all models or specific model on OAS dataset
        """
        from .Benchmarks.OAS_dataset import OAS_data_loader
        train_loader, test_loader, para_dict['seq_len'] = OAS_data_loader.OAS_data_loader(index_file='AIPT/Benchmarks/OAS_dataset/data/OAS_meta_info2.txt',
                                                      output_field='Species', input_type='full_length',
                                                      species_type=para_dict['species_type'], gapped=para_dict['gapped'],
                                                      pad=para_dict['pad'],
                                                      seq_dir='AIPT/Benchmarks/OAS_dataset/data/seq_db/')


        if args.model_name == 'Mason2020_CNN' or args.model_name == 'All':
            from .Benchmarks.OAS_dataset.comparison_OAS import Test_Mason2020_CNN
            Test_Mason2020_CNN(para_dict, train_loader, test_loader)

        if args.model_name == 'Mason2020_LSTM' or args.model_name == 'All':
            from .Benchmarks.OAS_dataset.comparison_OAS import Test_Mason2020_LSTM_RNN
            Test_Mason2020_LSTM_RNN(para_dict, train_loader, test_loader)

        if args.model_name == 'Wollacott2019':
            from .Benchmarks.OAS_dataset.comparison_OAS import Test_Wollacott2019_Bi_LSTM
            Test_Wollacott2019_Bi_LSTM(para_dict, train_loader, test_loader)
    else:
        print('Please provide the dataset name using the --dataset parameter')
    exit(0)


if __name__ is '__main__':
    main()