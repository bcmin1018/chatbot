from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle

def read_corpus_data(file_name):
    with open(file_name, 'rt', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data
# file_name = './corpus.txt'
# with open(file_name, 'rt', encoding='UTF8') as f:
#     for line in f.read().splitlines():
#         data = line.split('\t')
#         data = data[1:]


corpus_data = read_corpus_data('./corpus.txt')

p = Preprocess()
dict = []
for c in corpus_data:
    pos = p.pos(c[1])
    # print(pos)
    for k in pos:
        dict.append(k[0])

tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
tokenizer.fit_on_text(dict)
word_index = tokenzer.word_index

f = open('chatbot_dict.bin', 'wb')
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()