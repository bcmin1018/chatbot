# 모델 활용하는 부분

from kobert_transformers import get_tokenizer, get_kobert_model
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm_notebook
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
import warnings
import torch
import transformers
import numpy as np


# device = torch.device("cuda:0")
warnings.filterwarnings(action='ignore')

tokenizer = get_tokenizer()

###############학습 파라미터###############
TRAIN_BATCH_SIZE =16
VALID_BATCH_SIZE =16
EPOCHS = 5
LR = 1e-5
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.999
MODEL_PATH = "./intent_model.pt"
########################################

data = pd.read_csv("total_train_data.csv")
queries = data['query'].tolist()
intents = data['intent'].tolist()
train_data, valid_data, train_label, valid_label = train_test_split(queries, intents, test_size=0.2, shuffle=True, random_state=34)

from config.GlobalParams import MAX_LEN

train_dataset = BERT_Preprocess(texts=train_data,target=train_label, bert_tokenizer=tokenizer, max_len=MAX_LEN, pad=True, pair=False, mode = 'train')
train_data_loader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, num_workers = 2, shuffle=True)

valid_dataset = BERT_Preprocess(texts=valid_data,target=valid_label, bert_tokenizer=tokenizer, max_len=MAX_LEN, pad=True, pair=False, mode = 'train')
valid_data_loader = DataLoader(valid_dataset, batch_size = VALID_BATCH_SIZE, num_workers = 2, shuffle=True)

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

bert_model = get_kobert_model()
model = BERTClassifier(bert_model, dr_rate=0.2).to(device)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
loss_fn = nn.CrossEntropyLoss()
t_total = len(train_data_loader) * EPOCHS
warmup_step = int(t_total * 0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
max_grad_norm = 1
log_interval = 200

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


best_acc = 0
for e in range(EPOCHS):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, data in enumerate(tqdm_notebook(train_data_loader)):
        input_ids = data[0]['input_ids'].long().squeeze().to(device)
        token_type_ids = data[0]['token_type_ids'].long().squeeze().to(device)
        attention_mask = data[0]['attention_mask'].long().squeeze().to(device)
        labels = data[1].long().to(device)
        out = model(input_ids, token_type_ids, attention_mask)
        loss = loss_fn(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_acc += calc_accuracy(out, labels)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                     train_acc / (batch_id + 1)))
    print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))

    model.eval()
    for batch_id, data in enumerate(tqdm_notebook(valid_data_loader)):
        input_ids = data[0]['input_ids'].long().squeeze().to(device)
        token_type_ids = data[0]['token_type_ids'].long().squeeze().to(device)
        attention_mask = data[0]['attention_mask'].long().squeeze().to(device)
        labels = data[1].long().to(device)
        out = model(input_ids, token_type_ids, attention_mask)
        test_acc += calc_accuracy(out, labels)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({'epoch': e,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, MODEL_PATH)
    print("epoch {} validation acc {}".format(e + 1, test_acc / (batch_id + 1)))
