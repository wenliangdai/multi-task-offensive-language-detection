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
    parser.add_argument('-cl', '--clip', help='Using clip to gradients', action='store_true')
    parser.add_argument('-fr', '--freeze', help='Freeze the embedding layer or not to use less GPU memory', type=bool, required=False, default=False)
    parser.add_argument('-lw', '--loss-weights', help='Weights for all losses', nargs='+', type=float, required=False, default=[1, 1, 1, 1])

    # Transformers
    parser.add_argument('-ad', '--attention-dropout', help='transformer attention dropout', type=float, required=False, default=0.1)
    parser.add_argument('-hd', '--hidden-dropout', help='transformer hidden dropout', type=float, required=False, default=0.1)

    # LSTM
    parser.add_argument('-dr', '--dropout', help='dropout', type=float, required=False, default=0.1)
    parser.add_argument('-nl', '--num-layers', help='num of layers of LSTM', type=int, required=False, default=1)
    parser.add_argument('-hs', '--hidden-size', help='hidden vector size of LSTM', type=int, required=False, default=300)
    parser.add_argument('-hcm', '--hidden-combine-method', help='how to combbine hidden vectors in LSTM', type=str, required=False, default='concat')

    args = vars(parser.parse_args())
    return args
