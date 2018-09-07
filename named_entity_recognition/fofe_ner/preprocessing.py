# -*- coding: utf-8 -*-
"""
Preprocessors.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics.sequence_labeling import get_entities

UNK = '<UNK>'
PAD = '<PAD>'
NONE = '<NONE>'


class StaticPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, lowercase=True, num_norm=True):
        self._lowercase = lowercase
        self._num_norm = num_norm
        self.word_dic = {PAD: 0, UNK: 1}
        self.char_dic = {PAD: 0, UNK: 1}
        self.label_dic = {NONE: 0}

    def fit(self, X, y=None):
        for words in X:
            for w in words:
                for c in w:
                    if c in self.char_dic:
                        continue
                    self.char_dic[c] = len(self.char_dic)

                if self._lowercase:
                    w = w.lower()
                if w in self.word_dic:
                    continue
                self.word_dic[w] = len(self.word_dic)

        for tags in y:
            for t in tags:
                t = t.split('-')[-1]
                if t in self.label_dic:
                    continue
                self.label_dic[t] = len(self.label_dic)

        return self

    def transform(self, X, y=None):
        mentions = []
        mentions_char = []
        left_contexts = []
        right_contexts = []
        outputs = []
        for i, words in enumerate(X):
            char_ids = [self._get_char_ids(w) for w in words]
            word_ids = [self.word_dic.get(self.process(w), self.word_dic[UNK]) for w in words]
            ngrams = self.generate_ngrams(word_ids)
            if y is not None:
                d = {(begin_offset, end_offset + 1): t for t, begin_offset, end_offset in get_entities(y[i])}
            for l, r in ngrams:
                mentions.append(word_ids[l:r])
                mentions_char.append(char_ids[l:r])
                left_contexts.append(word_ids[:l])
                right_contexts.append(word_ids[r:])
                if y is not None and (l, r) in d:
                    outputs.append(self.label_dic[d[(l, r)]])
                else:
                    outputs.append(self.label_dic[NONE])

        outputs = np.array(outputs)

        inputs = [np.array(left_contexts), np.array(mentions), np.array(mentions_char), np.array(right_contexts)]

        return (inputs, outputs) if y is not None else inputs

    def generate_ngrams(self, words, n=7):
        res = []
        leng = len(words)
        for i in range(leng):
            for j in range(i + 1, min(i + 1 + n, leng + 1)):
                res.append((i, j))
        return res

    def process(self, w):
        if self._lowercase:
            w = w.lower()
        return w

    def _get_char_ids(self, word):
        return [self.char_dic.get(c, self.char_dic[UNK]) for c in word]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X=None, docs=None):
        if X is not None:
            id2char = {i: c for c, i in self.char_dic.items()}
            return [[id2char[c] for c in sent] for sent in X]
        id2label = {i: t for t, i in self.label_dic.items()}

        return [[id2label[t] for t in doc] for doc in docs]

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p


class DynamicPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, n_labels):
        self.n_labels = n_labels

    def transform(self, X, y=None):
        left_contexts, mentions, mentions_char, right_contexts = X
        mentions = pad_sequences(mentions, padding='post')
        mentions_char = pad_nested_sequences(mentions_char)
        left_contexts = pad_sequences(left_contexts, padding='pre')
        right_contexts = pad_sequences(right_contexts, padding='post')

        if y is not None:
            y = to_categorical(y, self.n_labels)
        sents = [left_contexts, mentions, mentions_char, right_contexts]

        return (sents, y) if y is not None else sents

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p


def pad_nested_sequences(sequences, dtype='int32'):
    """Pads nested sequences to the same length.

    This function transforms a list of list sequences
    into a 3D Numpy array of shape `(num_samples, max_sent_len, max_word_len)`.

    Args:
        sequences: List of lists of lists.
        dtype: Type of the output sequences.
    # Returns
        x: Numpy array.
    """
    max_sent_len = 0
    max_word_len = 0
    for sent in sequences:
        max_sent_len = max(len(sent), max_sent_len)
        for word in sent:
            max_word_len = max(len(word), max_word_len)

    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        for j, word in enumerate(sent):
            x[i, j, :len(word)] = word

    return x
