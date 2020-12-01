import torch
from torch import nn
from transformers import BertModel, RobertaModel
from .modules.attention import Attention

class MTL_Transformer_LSTM(nn.Module):
    def __init__(self, model, model_size, args):
        super(MTL_Transformer_LSTM, self).__init__()
        hidden_size = args['hidden_size']
        self.concat = args['hidden_combine_method'] == 'concat'
        input_size = 768 if model_size == 'base' else 1024

        if model == 'bert':
            MODEL = BertModel
            model_full_name = f'{model}-{model_size}-uncased'
        elif model == 'roberta':
            MODEL = RobertaModel
            model_full_name = f'{model}-{model_size}'

        self.emb = MODEL.from_pretrained(
            model_full_name,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        self.LSTMs = nn.ModuleDict({
            'a': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'b': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'c': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            )
        })

        self.attention_layers = nn.ModuleDict({
            'a': Attention(hidden_size * 2),
            'b': Attention(hidden_size * 2),
            'c': Attention(hidden_size * 2)
        })

        self.dropout = nn.Dropout(p=args['dropout'])

        linear_in_features = hidden_size * 2 if self.concat else hidden_size
        self.Linears = nn.ModuleDict({
            'a': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)
            ),
            'b': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 3)
            ),
            'c': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 4)
            )
        })

    def forward(self, inputs, lens, mask):
        embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)

        _, (h_a, _) = self.LSTMs['a'](embs)
        if self.concat:
            h_a = torch.cat((h_a[0], h_a[1]), dim=1)
        else:
            h_a = h_a[0] + h_a[1]
        h_a = self.dropout(h_a)

        _, (h_b, _) = self.LSTMs['b'](embs)
        if self.concat:
            h_b = torch.cat((h_b[0], h_b[1]), dim=1)
        else:
            h_b = h_b[0] + h_b[1]
        h_b = self.dropout(h_b)

        _, (h_c, _) = self.LSTMs['c'](embs)
        if self.concat:
            h_c = torch.cat((h_c[0], h_c[1]), dim=1)
        else:
            h_c = h_c[0] + h_c[1]
        h_c = self.dropout(h_c)

        logits_a = self.Linears['a'](h_a)
        logits_b = self.Linears['b'](h_b)
        logits_c = self.Linears['c'](h_c)

        return logits_a, logits_b, logits_c
