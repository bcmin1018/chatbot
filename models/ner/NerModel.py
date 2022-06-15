import tensorflow as tf
import numpy as np
from keras.models import Model, load_model
from keras.utils import pad_sequences

class NerModel:
    def __init__(self, model_name, preprocess):
        self.index_to_ner = {1: '0', 2: 'B_DT', 3: 'B_FOOD', 4: 'I', 5: 'B_OG', 6: 'B_PS',
                             8: 'NNP', 9: 'B_TI', 0: 'PAD'}
        self.model = load_model(model_name)
        self.p = preprocess

    def predict(self, query):
        pos = self.p.pos(query)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]
        max_len = 40
        padded_seqs = pad_sequences(sequences, padding='post', value=0, maxlen=max_len)
        predict = self.model.predict(np.array([padded_seqs[0]]))
        predict_class = tf.math.argmax(predict, axis=-1)
        tags = [self.index_to_ner[i] for i in predict_class.numpy()[0]]
        return list(zip(keywords, tags))

    def predict_tags(self, query):
        pos = self.p.pos(query)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]
        max_len = 40
        padded_seqs = pad_sequences(sequences, padding='post', value=0, maxlen=max_len)
        predict = self.model.predict(np.array([padded_seqs[0]]))
        predict_class = tf.math.argmax(predict, axis=-1)

        tags = []
        for tag_idx in predict_class.numpy()[0]:
            if tag_idx == 1: continue
            tags.append(self.index_to_ner[tag_idx])
            if len(tags) == 0:
                return None
            return tags