from torch import nn
from transformers import BertForSequenceClassification, RobertaForSequenceClassification

class BERT(nn.Module):
    def __init__(self, model_size, num_labels=2):
        super(BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(f'bert-{model_size}-uncased', num_labels=num_labels)

        # Freeze embeddings' parameters for saving memory
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False

    def forward(self, inputs, mask, labels):
        outputs = self.model(inputs, attention_mask=mask, labels=labels)
        loss, logits = outputs[:2]
        # return loss, logits
        return logits

class RoBERTa(nn.Module):
    def __init__(self, model_size, num_labels=2):
        super(RoBERTa, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(f'roberta-{model_size}', num_labels=num_labels)

        # Freeze embeddings' parameters for saving memory
        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False

    def forward(self, inputs, mask, labels):
        outputs = self.model(inputs, attention_mask=mask, labels=labels)
        loss, logits = outputs[:2]
        # return loss, logits
        return logits

class MTModel(nn.Module):
    def __init__(self, model, model_size):
        super(MTModel, self).__init__()
        if model == 'bert':
            pretrained = BertForSequenceClassification.from_pretrained(f'bert-{model_size}-uncased')
            self.main = pretrained.bert
            self.dropout = pretrained.dropout
        elif model == 'roberta':
            pretrained = RobertaForSequenceClassification.from_pretrained(f'roberta-{model_size}')
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
