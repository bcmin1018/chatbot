from konlpy.tag import Komoran
from kobert_transformers import get_tokenizer, get_kobert_model
tokenizer = get_tokenizer()
tokenizer()

class Preprocess:
    def __init__(self, userdic=None):
        self.komoran = Komoran(userdic=userdic)
        # 관계언, 기호, 어미, 접미사 제거
        self.exclusion_tags = [
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
        ]
    def pos(self, sentence):
        return self.komoran.pos(sentence)

    def get_keywords(self, pos, without_tag=False):
        f = lambda x: x in self.exclusion_tags
        word_list = []
        for p in pos:
            if f(p[1]) is False:
                word_list.append(p if without_tag is False else p[0])
        return word_list

class BERTDataset:
    def __init__(self, text, bert_tokenizer, max_len, pad, pair):
        self.text = text
        self.tokenizer = bert_tokenizer
        self.max_len = max_len
        self.transform = nlp.data.BERTSentenceTransform(self.tokenizer, max_seq_length=self.max_len, pad=pad, pair=pair)
        self.sentences = [self.transform([i]) for i in text]

    def __len__(self):
        return (len(self.labels))

    def __getitem__(self, i):
        return self.sentences[i]