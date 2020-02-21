import AIPT.Benchmarks.OAS_dataset.OAS_data_loader as OAS_data_loader
import AIPT.Models.Mason2020 as Mason2020
import AIPT.Models.Wollacott2019 as Wollacott2019
import numpy as np
import optparse

parser = optparse.OptionParser()

parser.add_option('--num-samples', type=int, default=1000)
parser.add_option('--seq-len', type=int, default=20)
parser.add_option('--batch-size', type=int, default=500)
parser.add_option('--dataset', type=str, default='OAS')
parser.add_option('--model-name', type=str, default='Mason2020_CNN')
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

if __name__ == '__main__':

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
                 'fixed_len': False,
                 'gapped': True,
                 'pad': True}

    if args.dataset == 'OAS':
        input_data = OAS_data_loader.OAS_data_loader(index_file='AIPT/Benchmarks/OAS_dataset/data/OAS_meta_info.txt',
                                                     output_field='Species', input_type='full_length',
                                                     species_type=['human'], gapped=para_dict['gapped'],
                                                     seq_dir='AIPT/Benchmarks/OAS_dataset/data/seq_db/')
        # in_data = input_data.data()
    else:
        print('Please provide the dataset')
        exit()

    if args.model_name == 'Mason2020_CNN':
        train_loader, test_loader, para_dict['seq_len'] = OAS_data_loader.create_loader(input_data,
                                                                                        pad=para_dict['pad'],
                                                                                        batch_size=args.batch_size,
                                                                                        gapped=para_dict['gapped'],
                                                                                        model_name=args.model_name)
        para_dict['model_name'] = 'CNN_Model'
        model = Mason2020.CNN.CNN_classifier(para_dict)
        model.fit(train_loader)
        output = model.predict(test_loader)
        labels = np.vstack([i for _, i in test_loader])
        mat, acc, mcc = model.evaluate(output, labels)

    elif args.model_name == 'Mason2020_LSTM_RNN':
        train_loader, test_loader, para_dict['seq_len'] = OAS_data_loader.create_loader(input_data,
                                                                                        pad=para_dict['pad'],
                                                                                        batch_size=args.batch_size,
                                                                                        gapped=para_dict['gapped'],
                                                                                        model_name=args.model_name)
        para_dict['hidden_dim'] = 40
        para_dict['model_name'] = 'LSTM_RNN'
        model = Mason2020.LSTM_RNN.LSTM_RNN_classifier(para_dict)
        model.fit(train_loader)
        output = model.predict(test_loader)
        labels = np.vstack([i for _, i in test_loader])
        mat, acc, mcc = model.evaluate(output, labels)

    elif args.model_name == 'Wollacott2019_Bi_LSTM':
        train_loader, test_loader, labels = OAS_data_loader.create_loader(input_data,
                                                                          pad=para_dict['pad'],
                                                                          batch_size=args.batch_size,
                                                                          gapped=para_dict['gapped'],
                                                                          model_name=args.model_name)
        para_dict['model_name'] = 'Bi_LSTM'
        model = Wollacott2019.Bi_LSTM.LSTM_Bi(para_dict)
        model.fit(train_loader)
        # output = model.predict(test_loader)
        # labels = np.concatenate([i for _, i in test_loader])
        # mat, acc, mcc = model.evaluate(output, labels)
        output = model.NLS_score(test_loader)
        # labels = np.vstack([i for _, i in test_loader])
        # print(labels)
        model.evaluate(output, labels)
