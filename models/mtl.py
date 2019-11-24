import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
from .modules.attention import Attention

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
            ),
            'final': nn.Sequential(
                nn.Linear(linear_in_features * 3, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)
            )
        })

        # Initialize weights with He initialization
        for l in self.Linears:
            self.Linears[l].apply(self.init_weights)

    def init_weights(layer):
        if type(layer) == nn.Linear:
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

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

        # _, (logits_a, _) = self.LSTMs['a'](embs)
        # _, (logits_b, _) = self.LSTMs['b'](embs)
        # _, (logits_c, _) = self.LSTMs['c'](embs)

        # logits_a = self.dropout(logits_a)
        # logits_b = self.dropout(logits_b)
        # logits_c = self.dropout(logits_c)

        # logits_a, _ = self.attention_layers['a'](output_a, lens)
        # logits_b, _ = self.attention_layers['b'](output_b, lens)
        # logits_c, _ = self.attention_layers['c'](output_c, lens)

        if not self.add_final:
            return logits_a, logits_b, logits_c
        else:
            # final_input = torch.cat((output_a, output_b, output_c), dim=2) # (batch, seq_len, num_directions * hidden_size * 3)
            # _, (logits_final, _) = self.LSTMs['final'](final_input)
            # if self.concat:
            #     logits_final = torch.cat((logits_final[0], logits_final[1]), dim=1)
            # else:
            #     logits_final = logits_final[0] + logits_final[1]
            # logits_final = self.Linears['final'](logits_final)
            # logits_final = self.dropout(logits_final)

            final_input = torch.cat((h_a, h_b, h_c), dim=1)
            logits_final = self.Linears['final'](final_input)
            return logits_a, logits_b, logits_c, logits_final


class MTL_Transformer_LSTM_gate(nn.Module):
    def __init__(self, model, model_size, args):
        super(MTL_Transformer_LSTM_gate, self).__init__()
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
                input_size=hidden_size * 2,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            )
        })

        linear_in_features = hidden_size * 2
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
            ),
            'final': nn.Sequential(
                nn.Linear(linear_in_features * 3, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)
            )
        })

        self.Linears.apply(self.init_weights)

        lstm_output_size = hidden_size * 2
        self.gates = nn.ModuleDict({
            'a': nn.Linear(lstm_output_size * 3, lstm_output_size),
            'b': nn.Linear(lstm_output_size * 3, lstm_output_size),
            'c': nn.Linear(lstm_output_size * 3, lstm_output_size)
        })

        self.gates.apply(self.init_weights)

    def init_weights(layer):
        if type(layer) == nn.Linear:
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

    def forward(self, inputs, lens, mask):
        embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)

        output_a, (h_a, _) = self.LSTMs['a'](embs)
        h_a = torch.cat((h_a[0], h_a[1]), dim=1)
        h_a = self.dropout(h_a)

        output_b, (h_b, _) = self.LSTMs['b'](embs)
        h_b = torch.cat((h_b[0], h_b[1]), dim=1)
        h_b = self.dropout(h_b)

        output_c, (h_c, _) = self.LSTMs['c'](embs)
        h_c = torch.cat((h_c[0], h_c[1]), dim=1)
        h_c = self.dropout(h_c)

        logits_a = self.Linears['a'](h_a)
        logits_b = self.Linears['b'](h_b)
        logits_c = self.Linears['c'](h_c)

        if not self.add_final:
            return logits_a, logits_b, logits_c

        # (batch, seq_len, num_directions * hidden_size * 3)
        gate_input = torch.cat((output_a, output_b, output_c), dim=2)
        gate_a = F.sigmoid(self.gates['a'](gate_input))
        gate_b = F.sigmoid(self.gates['b'](gate_input))
        gate_c = F.sigmoid(self.gates['c'](gate_input))
        final_input = gate_a * output_a + gate_b * output_b + gate_c * output_c
        _, (h_f, _) = self.LSTMs['final'](final_input)
        h_f = torch.cat((h_f[0], h_f[1]), dim=1)
        logits_f = self.Linears['final'](h_f)

        return logits_a, logits_b, logits_c, logits_f
