import torch
from torch import nn
from transformers import BertModel, BertForSequenceClassification, RobertaForSequenceClassification


class BERT(nn.Module):
    def __init__(self, model_size, args, num_labels=2):
        super(BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            f'bert-{model_size}-uncased',
            num_labels=num_labels,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        # Freeze embeddings' parameters for saving memory
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False

    def forward(self, inputs, mask, labels):
        outputs = self.model(inputs, attention_mask=mask, labels=labels)
        loss, logits = outputs[:2]
        # return loss, logits
        return logits

class RoBERTa(nn.Module):
    def __init__(self, model_size, args, num_labels=2):
        super(RoBERTa, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            f'roberta-{model_size}',
            num_labels=num_labels,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        # Freeze embeddings' parameters for saving memory
        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False

    def forward(self, inputs, mask, labels):
        outputs = self.model(inputs, attention_mask=mask, labels=labels)
        loss, logits = outputs[:2]
        # return loss, logits
        return logits

class MTModel(nn.Module):
    def __init__(self, model, model_size, args):
        super(MTModel, self).__init__()
        if model == 'bert':
            pretrained = BertForSequenceClassification.from_pretrained(
                f'bert-{model_size}-uncased',
                hidden_dropout_prob=args['hidden_dropout'],
                attention_probs_dropout_prob=args['attention_dropout']
            )
            self.main = pretrained.bert
            self.dropout = pretrained.dropout
        elif model == 'roberta':
            pretrained = RobertaForSequenceClassification.from_pretrained(
                f'roberta-{model_size}',
                hidden_dropout_prob=args['hidden_dropout'],
                attention_probs_dropout_prob=args['attention_dropout']
            )
            self.main = pretrained.roberta
            self.dropout = pretrained.dropout

        # Freeze embeddings' parameters for saving memory
        for param in self.main.embeddings.parameters():
            param.requires_grad = False

        linear_in_features = 768 if model_size == 'base' else 1024

        self.classifier_a = nn.Linear(in_features=linear_in_features, out_features=2, bias=True)
        self.classifier_b = nn.Linear(in_features=linear_in_features, out_features=3, bias=True)
        self.classifier_c = nn.Linear(in_features=linear_in_features, out_features=4, bias=True)

    def forward(self, inputs, mask):
        outputs = self.main(inputs, attention_mask=mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        # logits for 3 sub-tasks
        logits_A = self.classifier_a(pooled_output)
        logits_B = self.classifier_b(pooled_output)
        logits_C = self.classifier_c(pooled_output)
        return logits_A, logits_B, logits_C

class BERT_LSTM(nn.Module):
    def __init__(self, model_size, num_labels, args):
        super(BERT_LSTM, self).__init__()
        hidden_size = args['hidden_size']
        self.concat = args['hidden_combine_method'] == 'concat'
        input_size = 768 if model_size == 'base' else 1024

        self.emb = BertModel.from_pretrained(
            f'bert-{model_size}-uncased',
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=args['num_layers'],
            bidirectional=True,
            batch_first=True,
            dropout=args['dropout'] if args['num_layers'] > 1 else 0
        )
        self.dropout = nn.Dropout(p=args['dropout'])
        self.linear = nn.Linear(in_features=hidden_size * 2 if self.concat else hidden_size, out_features=num_labels)

    def forward(self, inputs, mask, labels):
        embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)
        _, (h_n, _) = self.lstm(input=embs) # (num_layers * num_directions, batch, hidden_size)
        if self.concat:
            h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        else:
            h_n = h_n[0] + h_n[1]
        h_n = self.dropout(h_n)
        logits = self.linear(h_n)
        return logits

# class BERT_LSTM_MTL(nn.Module):
#     def __init__(self, model, model_size, args, input_size=768):
#         super(BERT_LSTM_MTL, self).__init__()
#         hidden_size = args['hidden_size']

#         self.emb = BertModel.from_pretrained(
#             f'bert-{model_size}-uncased',
#             hidden_dropout_prob=args['hidden_dropout'],
#             attention_probs_dropout_prob=args['attention_dropout']
#         )
#         # self.main = pretrained.bert
#         # self.dropout = pretrained.dropout
#         self.LSTMs = nn.ModuleDict({
#             'a': nn.LSTM(
#                 input_size=input_size,
#                 hidden_size=hidden_size,
#                 num_layers=args['num_layers'],
#                 bidirectional=True,
#                 batch_first=True,
#                 dropout=args['dropout'] if args['num_layers'] > 1 else 0
#             ),
#             'b': nn.LSTM(
#                 input_size=input_size,
#                 hidden_size=hidden_size,
#                 num_layers=args['num_layers'],
#                 bidirectional=True,
#                 batch_first=True,
#                 dropout=args['dropout'] if args['num_layers'] > 1 else 0
#             ),
#             'c': nn.LSTM(
#                 input_size=input_size,
#                 hidden_size=hidden_size,
#                 num_layers=args['num_layers'],
#                 bidirectional=True,
#                 batch_first=True,
#                 dropout=args['dropout'] if args['num_layers'] > 1 else 0
#             )
#         })
#         self.dropout = nn.Dropout(p=args['dropout'])
#         self.Linears = nn.ModuleDict({
#             'a': nn.Linear(in_features=hidden_size * 2, out_features=2),
#             'b': nn.Linear(in_features=hidden_size * 2, out_features=3),
#             'c': nn.Linear(in_features=hidden_size * 2, out_features=4)
#         })

#     def forward(self, inputs, mask):
#         embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)

#         _, (logits_a, _) = self.LSTMs['a'](embs)
#         logits_a = torch.cat((logits_a[0], logits_a[1]), dim=1)
#         logits_a = self.dropout(logits_a)
#         logits_a = self.Linears['a'](logits_a)

#         _, (logits_b, _) = self.LSTMs['b'](embs)
#         logits_b = torch.cat((logits_b[0], logits_b[1]), dim=1)
#         logits_b = self.dropout(logits_b)
#         logits_b = self.Linears['b'](logits_b)

#         _, (logits_c, _) = self.LSTMs['c'](embs)
#         logits_c = torch.cat((logits_c[0], logits_c[1]), dim=1)
#         logits_c = self.dropout(logits_c)
#         logits_c = self.Linears['c'](logits_c)

#         return logits_a, logits_b, logits_c

