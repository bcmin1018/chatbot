from transformers import BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, tqdm_notebook
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import transformers
import pandas as pd
import numpy as np
import torch
import random
import warnings
import gluonnlp as nlp


train_file = 'total_train_data.csv'
data = pd.read_csv(train_file)
queries = data['query'].tolist()
intents = data['intent'].tolist()


# bertmodel, vocab = get_pytorch_kobert_model()
# tokenizer = get_tokenizer()
# tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)