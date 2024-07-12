import torch
import torch.nn as nn
from transformers import BertModel

class CustomModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(CustomModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]
        out = self.classifier(cls_output)
        return out
