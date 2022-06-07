# 모델 활용하는 부분
import torch
from torch import nn
from torch.nn import functional as F
from utils.Preprocess import BERTDataset
from kobert_tokenizer import KoBERTTokenizer

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
        self.query = BERTDataset(text=query, bert_tokenizer=tok, max_len=MAX_LEN, pad=True,
                                    pair=False)
        self.model_load.eval()
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))