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

        self.classifier_a = nn.Linear(in_features=768, out_features=2, bias=True)
        self.classifier_b = nn.Linear(in_features=768, out_features=3, bias=True)
        self.classifier_c = nn.Linear(in_features=768, out_features=4, bias=True)

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
    def __init__(self, model_size, num_labels, args, input_size=768):
        super(BERT_LSTM, self).__init__()
        hidden_size = args['hidden_size']

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
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=num_labels)

    def forward(self, inputs, mask, labels):
        embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)
        _, (h_n, _) = self.lstm(input=embs) # (num_layers * num_directions, batch, hidden_size)
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        h_n = self.dropout(h_n)
        logits = self.linear(h_n)
        return logits

class BERT_LSTM_MTL(nn.Module):
    def __init__(self, model, model_size, args, input_size=768):
        super(BERT_LSTM_MTL, self).__init__()
        hidden_size = args['hidden_size']

        self.emb = BertModel.from_pretrained(
            f'bert-{model_size}-uncased',
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )
        # self.main = pretrained.bert
        # self.dropout = pretrained.dropout
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
        self.dropout = nn.Dropout(p=args['dropout'])
        self.Linears = nn.ModuleDict({
            'a': nn.Linear(in_features=hidden_size * 2, out_features=2),
            'b': nn.Linear(in_features=hidden_size * 2, out_features=3),
            'c': nn.Linear(in_features=hidden_size * 2, out_features=4)
        })

    def forward(self, inputs, mask):
        embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)

        _, (logits_a, _) = self.LSTMs['a'](embs)
        logits_a = torch.cat((logits_a[0], logits_a[1]), dim=1)
        logits_a = self.dropout(logits_a)
        logits_a = self.Linears['a'](logits_a)

        _, (logits_b, _) = self.LSTMs['b'](embs)
        logits_b = torch.cat((logits_b[0], logits_b[1]), dim=1)
        logits_b = self.dropout(logits_b)
        logits_b = self.Linears['b'](logits_b)

        _, (logits_c, _) = self.LSTMs['c'](embs)
        logits_c = torch.cat((logits_c[0], logits_c[1]), dim=1)
        logits_c = self.dropout(logits_c)
        logits_c = self.Linears['c'](logits_c)

        return logits_a, logits_b, logits_c

class GatedModel(nn.Module):
    def __init__(self, model, model_size, args):
        super(GatedModel, self).__init__()
        # using BERT/RoBERTa pre-trained model
        if model == 'bert':
            pretrainedA = BertForSequenceClassification.from_pretrained(f'bert-{model_size}-uncased')
            self.mainA = pretrainedA.bert
            pretrainedB = BertForSequenceClassification.from_pretrained(f'bert-{model_size}-uncased')
            self.mainB = pretrainedB.bert
            pretrainedC = BertForSequenceClassification.from_pretrained(f'bert-{model_size}-uncased')
            self.mainC = pretrainedC.bert

            # Freeze embeddings' parameters for saving memory
            for p in [
                # *self.model.robe
                *self.mainA.embeddings.word_embeddings.parameters(),
                *self.mainB.embeddings.word_embeddings.parameters(),
                *self.mainC.embeddings.word_embeddings.parameters(),
            ]:
                p.requires_grad = False

            self.dropoutA = pretrainedA.dropout
            self.dropoutB = pretrainedB.dropout
            self.dropoutC = pretrainedC.dropout
            if model_size == 'base':
                self.hidden_size = 768
            if model_size == 'large':
                self.hidden_size = 1024

            self.linearA = nn.Linear(in_features=self.hidden_size*3, out_features=self.hidden_size, bias=True)
            self.linearB = nn.Linear(in_features=self.hidden_size*3, out_features=self.hidden_size, bias=True)
            self.linearC = nn.Linear(in_features=self.hidden_size*3, out_features=self.hidden_size, bias=True)

            self.softmaxA = nn.Softmax(dim=1)
            self.softmaxB = nn.Softmax(dim=1)
            self.softmaxC = nn.Softmax(dim=1)

            self.classifier_a = nn.Linear(in_features=self.hidden_size, out_features=2, bias=True)
            self.classifier_b = nn.Linear(in_features=self.hidden_size, out_features=3, bias=True)
            self.classifier_c = nn.Linear(in_features=self.hidden_size, out_features=4, bias=True)

        elif model == 'roberta':
            pretrainedA = RobertaForSequenceClassification.from_pretrained(f'roberta-{model_size}')
            self.mainA = pretrainedA.roberta
            pretrainedB = RobertaForSequenceClassification.from_pretrained(f'roberta-{model_size}')
            self.mainB = pretrainedB.roberta
            pretrainedC = RobertaForSequenceClassification.from_pretrained(f'roberta-{model_size}')
            self.mainC = pretrainedC.roberta

            # Freeze embeddings' parameters for saving memory
            for p in [
                # *self.model.robe
                *self.mainA.embeddings.word_embeddings.parameters(),
                *self.mainB.embeddings.word_embeddings.parameters(),
                *self.mainC.embeddings.word_embeddings.parameters(),
            ]:
                p.requires_grad = False

            self.dropoutA = pretrainedA.classifier
            self.dropoutB = pretrainedB.classifier
            self.dropoutC = pretrainedC.classifier

            if model_size == 'base':
                self.hidden_size = 768
            if model_size == 'large':
                self.hidden_size = 1024

            self.linearA = nn.Linear(in_features=self.hidden_size*3, out_features=self.hidden_size, bias=True)
            self.linearB = nn.Linear(in_features=self.hidden_size*3, out_features=self.hidden_size, bias=True)
            self.linearC = nn.Linear(in_features=self.hidden_size*3, out_features=self.hidden_size, bias=True)

            self.softmaxA = nn.Softmax(dim=1)
            self.softmaxB = nn.Softmax(dim=1)
            self.softmaxC = nn.Softmax(dim=1)

            self.classifier_a = nn.Linear(in_features=self.hidden_size, out_features=2, bias=True)
            self.classifier_b = nn.Linear(in_features=self.hidden_size, out_features=3, bias=True)
            self.classifier_c = nn.Linear(in_features=self.hidden_size, out_features=4, bias=True)

    def forward(self, inputs, mask):
        outputsA = self.mainA(inputs, attention_mask=mask)
        pooled_outputA = outputsA[1]
        # pooled_outputA = self.dropoutA(pooled_outputA) # batch_size * hidden_size

        outputsB = self.mainB(inputs, attention_mask=mask)
        pooled_outputB = outputsB[1]
        # pooled_outputB = self.dropoutB(pooled_outputB) # batch_size * hidden_size

        outputsC = self.mainC(inputs, attention_mask=mask)
        pooled_outputC = outputsC[1]
        # pooled_outputC = self.dropoutC(pooled_outputC) # batch_size * hidden_size

        gateA = self.softmaxA(self.linearA(torch.cat((pooled_outputA, pooled_outputB, pooled_outputC), 1)))
        gateB = self.softmaxB(self.linearB(torch.cat((pooled_outputA, pooled_outputB, pooled_outputC), 1)))
        gateC = self.softmaxC(self.linearC(torch.cat((pooled_outputA, pooled_outputB, pooled_outputC), 1)))

        gateA_1 = torch.ones(gateA.shape).cuda() - gateA
        gateB_1 = torch.ones(gateB.shape).cuda() - gateB
        gateC_1 = torch.ones(gateC.shape).cuda() - gateC

        hidden_A = torch.mul(gateA_1, pooled_outputA) + torch.mul(torch.mul(gateA, 0.5), pooled_outputB) + torch.mul(torch.mul(gateA, 0.5), pooled_outputC)
        hidden_B = torch.mul(gateB_1, pooled_outputA) + torch.mul(torch.mul(gateB, 0.5), pooled_outputB) + torch.mul(torch.mul(gateB, 0.5), pooled_outputC)
        hidden_C = torch.mul(gateC_1, pooled_outputA) + torch.mul(torch.mul(gateC, 0.5), pooled_outputB) + torch.mul(torch.mul(gateC, 0.5), pooled_outputC)

        # logits for 3 sub-tasks
        logits_A = self.classifier_a(hidden_A)
        logits_B = self.classifier_b(hidden_B)
        logits_C = self.classifier_c(hidden_C)

        return logits_A, logits_B, logits_C
