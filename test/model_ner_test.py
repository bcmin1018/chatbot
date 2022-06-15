from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel

p = Preprocess(word2index_dic='../train_tools/dict/chatbot_dict.bin', userdic='../utils/user_dict.tsv')

ner = NerModel(model_name='../models/ner/ner_model.h5', preprocess=p)
query = '오늘 오전 13시 2분에 탕수육 주문하고 싶어요'
predicts = ner.predict(query)
print(predicts)