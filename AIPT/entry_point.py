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
    parser.add_option('--batch-size', type=int, default=5)
    parser.add_option('--dataset', type=str, default='Multitask')
    parser.add_option('--model-name', type=str, default='CNN')
    parser.add_option('--optim-name', type=str, default='Adam')
    parser.add_option('--epoch', type=int, default=2)
    parser.add_option('--learning-rate', type=float, default=1e-3)
    parser.add_option('--step-size', type=int, default=10)
    parser.add_option('--dropout-rate', type=float, default=0.5)
    parser.add_option('--random-state', type=int, default=100)  # For splitting the data

    parser.add_option('--n-filter', type=int, default=400)  # CNN model
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
                 'n-filter': args.n_filter,
                 'filter_size': args.filter_size,
                 'fc_hidden_dim': args.fc_hidden_dim,
                 'dropout_rate': args.dropout_rate,
                 'hidden_layer_num': args.hidden_layer_num,
                 'hidden_dim': args.hidden_dim,
                 'embedding_dim': args.embedding_dim,
                 'species_type': ['human','mouse','rabbit','rhesus'],
                 'cells_type': ['Memory-B-Cells', 'Naive-B-Cells'],
                 'Multitask': [('Species',['human','mouse','rabbit','rhesus']), ('BType',['Memory-B-Cells', 'Naive-B-Cells'])],
                 'fixed_len': False,
                 'gapped': True,
                 'pad': True} # padding False for Wollacott model

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
        train_loader, train_eval_loader, test_eval_loader, para_dict['seq_len'] = OAS_data_loader.OAS_data_loader(index_file='AIPT/Benchmarks/OAS_dataset/data/OAS_meta_info.txt',
                                                      output_field='Species', input_type='CDR3_full', batch_size=para_dict['batch_size'],
                                                      species_type=para_dict['species_type'], gapped=para_dict['gapped'],
                                                      pad=para_dict['pad'], model_name= args.model_name,
                                                      seq_dir='AIPT/Benchmarks/OAS_dataset/data/seq_db/')


        if args.model_name == 'Mason2020_CNN' or args.model_name == 'All':
            from .Benchmarks.OAS_dataset.comparison_OAS import Test_Mason2020_CNN
            Test_Mason2020_CNN(para_dict, train_loader, train_eval_loader, test_eval_loader)

        if args.model_name == 'Mason2020_LSTM' or args.model_name == 'All':
            from .Benchmarks.OAS_dataset.comparison_OAS import Test_Mason2020_LSTM_RNN
            Test_Mason2020_LSTM_RNN(para_dict, train_loader, train_eval_loader, test_eval_loader)

        if args.model_name == 'Wollacott2019':
            from .Benchmarks.OAS_dataset.comparison_OAS import Test_Wollacott2019_Bi_LSTM
            Test_Wollacott2019_Bi_LSTM(para_dict, train_loader, train_eval_loader, test_eval_loader)

    elif args.dataset == 'Benchmark':

        from AIPT.Benchmarks.OAS_dataset.Benchmark import OAS_data_loader
        train_loader, train_eval_loader, test_eval_loader, para_dict['seq_len'] = OAS_data_loader(
                                index_file='AIPT/Benchmarks/OAS_dataset/data/OAS_meta_info_temp.txt',
                                output_field='Species', input_type='full_length',
                                species_type=para_dict['species_type'], gapped=para_dict['gapped'],
                                pad=para_dict['pad'], model_name=args.model_name,
                                seq_dir='AIPT/Benchmarks/OAS_dataset/data/seq_db/')

        if args.model_name == 'Wollacott2019':
            from .Benchmarks.OAS_dataset.comparison_OAS import Benchmark_Wollacott2019
            Benchmark_Wollacott2019(para_dict, train_loader, train_eval_loader, test_eval_loader)

    elif args.dataset == 'Multitask':

        from AIPT.Benchmarks.OAS_dataset import Multitask_learning as ML
        from AIPT.Utils.model import Model

        if args.model_name == 'CNN':

            train_loader, train_eval_loader, test_eval_loader, para_dict['seq_len'] = ML.OAS_data_loader(
                index_file='AIPT/Benchmarks/OAS_dataset/data/OAS_meta_info.txt',
                output_field=[para_dict['Multitask'][i][0] for i in range(len(para_dict['Multitask']))],  input_type='CDR3',
                species_type=para_dict['Multitask'], gapped=para_dict['gapped'],
                pad=para_dict['pad'], model_name=args.model_name,
                seq_dir='AIPT/Benchmarks/OAS_dataset/data/seq_db/')

            para_dict['model_name'] = 'Multitask_CNN'
            print([para_dict['Multitask'][i][0] for i in range(len(para_dict['Multitask']))])
            para_dict['num_classes'] = [len(para_dict['Multitask'][i][1]) for i in range(len(para_dict['Multitask']))]
            print('Parameters: ', para_dict)
            model = ML.Multitask_CNN(para_dict)
            model.fit(train_loader)
            # print('Training_evaluation')
            # output = Model.predict(model, train_eval_loader)
            # labels = [i for _, i in test_eval_loader]
            # model.evaluate(output, labels)
            print('Test data evaluation')
            outputs = Model.predict(model, test_eval_loader)
            labels = [i for _, i in test_eval_loader]
            model.evaluate(outputs, labels, para_dict)

        elif args.model_name == 'LSTM_RNN':

            train_loader, train_eval_loader, test_eval_loader, para_dict['seq_len'] = ML.OAS_data_loader(
                index_file='AIPT/Benchmarks/OAS_dataset/data/OAS_meta_info.txt',
                output_field=[para_dict['Multitask'][i][0] for i in range(len(para_dict['Multitask']))],  input_type='CDR3',
                species_type=para_dict['Multitask'], gapped=para_dict['gapped'],
                pad=para_dict['pad'], model_name=args.model_name,
                seq_dir='AIPT/Benchmarks/OAS_dataset/data/seq_db/')

            para_dict['model_name'] = 'Multitask_LSTM'
            para_dict['num_classes'] = [len(para_dict['Multitask'][i][1]) for i in range(len(para_dict['Multitask']))]
            print('Parameters: ', para_dict)
            model = ML.Multitask_LSTM_RNN(para_dict)
            model.fit(train_loader)
            # print('Training_evaluation')
            # output = Model.predict(model, train_eval_loader)
            # labels = [i for _, i in train_eval_loader]
            # model.evaluate(output, labels, para_dict)
            print('Test data evaluation')
            outputs = Model.predict(model, test_eval_loader)
            labels = [i for _, i in test_eval_loader]
            model.evaluate(outputs, labels, para_dict)

        elif args.model_name == 'Bi_LSTM':

            train_loader, train_eval_loader, test_eval_loader, para_dict['seq_len'] = ML.OAS_data_loader(
                index_file='AIPT/Benchmarks/OAS_dataset/data/OAS_meta_info.txt',
                output_field=[para_dict['Multitask'][i][0] for i in range(len(para_dict['Multitask']))],  input_type='CDR3',
                species_type=para_dict['Multitask'], gapped=para_dict['gapped'],
                pad=para_dict['pad'], model_name=args.model_name,
                seq_dir='AIPT/Benchmarks/OAS_dataset/data/seq_db/')

            para_dict['model_name'] = 'Multitask_Bi_LSTM'
            para_dict['num_classes'] = [len(para_dict['Multitask'][i][1]) for i in range(len(para_dict['Multitask']))]
            print('Parameters: ', para_dict)
            model = ML.Multitask_Bi_LSTM(para_dict)
            model.fit(train_loader)
            # print('Training_evaluation')
            # output = Model.predict(model, train_eval_loader)
            # labels = [i for _, i in test_eval_loader]
            # model.evaluate(output, labels)
            print('Test data evaluation')
            outputs = Model.predict(model, test_eval_loader)
            labels = [i for _, i in test_eval_loader]
            model.evaluate(outputs, labels, para_dict)

    elif args.dataset == 'Naive_Memory_cells':

        if args.model_name == 'Mason2020_CNN' or args.model_name == 'All':
            
            from .Models.Mason2020 import CNN
            from .Benchmarks.OAS_dataset import OAS_data_loader
            train_loader, train_eval_loader, test_eval_loader, para_dict['seq_len'] = OAS_data_loader.OAS_data_loader_for_memory_naive_cells(
                                index_file=args.index_file,
                                output_field='BType', input_type='full_length',
                                species_type=para_dict['cells_type'], gapped=para_dict['gapped'],
                                pad=para_dict['pad'], model_name=args.model_name,
                                seq_dir=args.seq_dir)
            para_dict['model_name'] = 'CNN_Model'
            para_dict['num_classes'] = len(para_dict['cells_type'])
            print('Parameters: ', para_dict)
            model = CNN.CNN_classifier(para_dict)
            model.fit(train_loader)
#             print('Training_evaluation')
#             output = model.predict(train_eval_loader)
#             labels = np.vstack([i for _, i in train_eval_loader])
#             model.evaluate('Train', output, labels)
            print('Test data evaluation')
            output = model.predict(test_eval_loader)
            labels = np.vstack([i for _, i in test_eval_loader])
            model.evaluate('Test', output, labels)
            
        if args.model_name == 'Mason2020_LSTM' or args.model_name == 'All':
            
            from .Models.Mason2020 import LSTM_RNN
            from .Benchmarks.OAS_dataset import OAS_data_loader
            train_loader, train_eval_loader, test_eval_loader, para_dict['seq_len'] = OAS_data_loader.OAS_data_loader_for_memory_naive_cells(
                                index_file=args.index_file,
                                output_field='BType', input_type='full_length',
                                species_type=para_dict['cells_type'], gapped=para_dict['gapped'],
                                pad=para_dict['pad'], model_name=args.model_name,
                                seq_dir=args.seq_dir)
            para_dict['model_name'] = 'LSTM_RNN_classifier'
            para_dict['num_classes'] = len(para_dict['cells_type'])
            print('Parameters: ', para_dict)
            model = LSTM_RNN.LSTM_RNN_classifier(para_dict)
            model.fit(train_loader)
#             print('Training_evaluation')
#             output = model.predict(train_eval_loader)
#             labels = np.vstack([i for _, i in train_eval_loader])
#             model.evaluate('Train', output, labels)
            print('Test data evaluation')
            output = model.predict(test_eval_loader)
            labels = np.vstack([i for _, i in test_eval_loader])
            model.evaluate('Test', output, labels)
            
        if args.model_name == 'Wollacott2019':
            from .Benchmarks.benchmark import OAS_data_loader_for_memory_naive_cells
            train_loader, train_eval_loader, test_eval_loader, para_dict['seq_len'] = OAS_data_loader_for_memory_naive_cells(
                                index_file=args.index_file,
                                output_field='BType', input_type='full_length',
                                species_type=para_dict['cells_type'], gapped=para_dict['gapped'],
                                pad=para_dict['pad'], model_name=args.model_name,
                                seq_dir=args.seq_dir)
            from .Benchmarks.benchmark import Benchmark
            para_dict['model_name'] = 'Benchmark_Wollacott2019'
            para_dict['num_classes'] = len(para_dict['cells_type'])
            print('Parameters: ', para_dict)
            model = Benchmark(para_dict)
            model.fit(train_loader)
#             print('Train data evaluation')
#             output = model.predict(train_eval_loader)
#             labels = np.vstack([i for _, i in train_eval_loader])
#             model.evaluate('Train', output, labels)
            print('Test data evaluation')
            output = model.predict(test_eval_loader)
            labels = np.vstack([i for _, i in test_eval_loader])
            model.evaluate('Test', output, labels)


    else:
        print('Please provide the dataset name using the --dataset parameter')
    exit(0)


if __name__ is '__main__':
    main()