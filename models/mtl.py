import torch
from torch import nn
from transformers import BertModel, RobertaModel

class MTL_Transformer_LSTM(nn.Module):
    def __init__(self, model, model_size, args):
        super(MTL_Transformer_LSTM, self).__init__()
        hidden_size = args['hidden_size']
        self.add_final = args['add_final']

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
            ),
            'final': nn.LSTM(
                input_size=hidden_size * 6,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=False,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            )
        })
        self.dropout = nn.Dropout(p=args['dropout'])
        linear_in_features = hidden_size * 2 if self.concat else hidden_size
        self.Linears = nn.ModuleDict({
            'a': nn.Linear(in_features=linear_in_features, out_features=2),
            'b': nn.Linear(in_features=linear_in_features, out_features=3),
            'c': nn.Linear(in_features=linear_in_features, out_features=4),
            'final': nn.Linear(in_features=linear_in_features, out_features=2)
        })

    def forward(self, inputs, mask):
        embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)

        output_a, (logits_a, _) = self.LSTMs['a'](embs)
        if self.concat:
            logits_a = torch.cat((logits_a[0], logits_a[1]), dim=1)
        else:
            logits_a = logits_a[0] + logits_a[1]
        logits_a = self.dropout(logits_a)

        output_b, (logits_b, _) = self.LSTMs['b'](embs)
        if self.concat:
            logits_b = torch.cat((logits_b[0], logits_b[1]), dim=1)
        else:
            logits_b = logits_b[0] + logits_b[1]
        logits_b = self.dropout(logits_b)

        output_c, (logits_c, _) = self.LSTMs['c'](embs)
        if self.concat:
            logits_c = torch.cat((logits_c[0], logits_c[1]), dim=1)
        else:
            logits_c = logits_c[0] + logits_c[1]
        logits_c = self.dropout(logits_c)

        logits_a = self.Linears['a'](logits_a)
        logits_b = self.Linears['b'](logits_b)
        logits_c = self.Linears['c'](logits_c)

        if not self.add_final:
            return logits_a, logits_b, logits_c
        else:
            final_input = torch.cat((output_a, output_b, output_c), dim=2) # (seq_len, batch, num_directions * hidden_size * 3)
            _, (logits_final, _) = self.LSTMs['final'](final_input)
            if self.concat:
                logits_final = torch.cat((logits_final[0], logits_final[1]), dim=1)
            else:
                logits_final = logits_final[0] + logits_final[1]
            logits_final = self.Linears['final'](logits_final)
            return logits_a, logits_b, logits_c, logits_final
