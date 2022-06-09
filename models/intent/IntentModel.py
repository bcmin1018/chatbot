from kobert_transformers import get_tokenizer
from config.GlobalParams import MAX_LEN
from torch import nn
import torch

class IntentModel:
    def __init__(self, bert_model, model_path):
        self.labels = {0: "인사", 1: "욕설", 2: "주문", 3: "예약", 4: "기타"}
        self.model = BERTClassifier(bert_model)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
        self.model.eval()
        self.tokenizer = get_tokenizer()

    def predict_class(self, query):
        transform = self.tokenizer(query,
                            padding="max_length",
                            max_length = MAX_LEN,
                            truncation=True,
                            return_tensors='pt',
                            add_special_tokens=True)

        with torch.no_grad():
            output = self.model(transform['input_ids'], transform['token_type_ids'], transform['attention_mask'])
            predict = torch.argmax(output, 1).cpu().detach().numpy()
        return predict[0]


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size = 768, num_classes=5, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)


    def forward(self, token_ids, segment_ids, attention_mask):
        pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids, attention_mask = attention_mask)[1]
        if self.dr_rate:
            out = self.dropout(pooler)
            return self.classifier(out)
        else:
            return self.classifier(pooler)