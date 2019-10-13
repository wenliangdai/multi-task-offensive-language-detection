from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification

class BERT_BASE(nn.Module):
    def __init__(self, pretrained_type='bert-base-uncased'):
        super(BERT_BASE, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_type)
        self.model = BertForSequenceClassification.from_pretrained(pretrained_type, num_labels=2)

    def forward(self, inputs, mask, labels):
        outputs = self.model(inputs, attention_mask=mask, labels=labels)
        loss, logits = outputs[:2]
        return loss, logits
