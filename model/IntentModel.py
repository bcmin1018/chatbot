# 모델 활용하는 부분
import torch
from torch import nn
from torch.nn import functional as F

class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=5, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class IntentModel:
    def __init__(self, model_name, model_path):
        self.labels = {0: "인사", 1: "욕설", 2: "주문", 3: "예약", 4: "기타"}
        self.model = BERTClassifier(dr_rate=0.2)
        self.model_path = model_path
        self.model_load = self.model.load_state_dict(torch.load(self.model_path))

    def predict_class(self, query):