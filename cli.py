import argparse

def get_args():
    parser = argparse.ArgumentParser(description='TRADE Multi-Domain DST')

    # Training hyper-parameters
    parser.add_argument('-bs', '--batch_size', help='Batch size', type=int, required=True)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', type=float, required=True)
    parser.add_argument('-ep', '--epochs', help='Number of epochs', type=int, required=True)
    parser.add_argument('-tn', '--task_name', help='Task name (no space E.g. task2_bert), used for suffix of filenames', type=str, required=True)
    parser.add_argument('-pa', '--patience', help='Patience to stop training', type=int, required=False, default=5)
    parser.add_argument('-pr', '--print_iter', help='Print every X iterations during training', type=int, required=False, default=100)
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')

    args = vars(parser.parse_args())
    return args
