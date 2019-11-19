import argparse

def get_args():
    parser = argparse.ArgumentParser(description='TRADE Multi-Domain DST')

    # Training hyper-parameters
    parser.add_argument('-bs', '--batch_size', help='Batch size', type=int, required=True)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', type=float, required=True)
    parser.add_argument('-wd', '--weight_decay', help='Weight decay', type=float, required=False, default=0)
    parser.add_argument('-ep', '--epochs', help='Number of epochs', type=int, required=True)
    parser.add_argument('-tr', '--truncate', help='Truncate the sequence length to', type=int, required=False, default=-1)
    parser.add_argument('-pa', '--patience', help='Patience to stop training', type=int, required=False, default=5)
    parser.add_argument('-pr', '--print_iter', help='Print every X iterations during training', type=int, required=False, default=100)
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')
    parser.add_argument('-ta', '--task', help='Which subtask to run', type=str, required=True)
    parser.add_argument('-mo', '--model', help='Which model to use', type=str, required=True)
    parser.add_argument('-ms', '--model_size', help='Which size of model to use', type=str, required=False, default='base')
    parser.add_argument('-cl', '--clip', help='Using clip to gradients', type=bool, required=False, default=False)
    parser.add_argument('-fr', '--freeze', help='Freeze the embedding layer or not to use less GPU memory', type=bool, required=False, default=False)

    args = vars(parser.parse_args())
    return args
