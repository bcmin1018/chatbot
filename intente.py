import transformers
import pandas as pd
import numpy as np
import torch
import random
import warnings
import gluonnlp as nlp
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, tqdm_notebook

class IntentDataset:
    def __init__(self, text, target, bert_tokenizer, max_len, pad, pair):
        self.text = text
        self.target = target
        self.tokenizer = bert_tokenizer
        self.max_len = max_len
        self.transform = nlp.data.BERTSentenceTransform(self.tokenizer, max_seq_length=self.max_len, pad=pad, pair=pair)
        self.sentences = [self.transform([i]) for i in text]
        self.labels = [np.int32(i) for i in target]

    def __len__(self):
        return (len(self.labels))

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))