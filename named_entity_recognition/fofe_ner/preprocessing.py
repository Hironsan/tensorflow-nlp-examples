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
from nltk import ngrams

from .utils import Vocabulary


class IndexTransformer(BaseEstimator, TransformerMixin):
    """Convert a collection of raw documents to a document id matrix.

    Attributes:
        _use_char: boolean. Whether to use char feature.
        _num_norm: boolean. Whether to normalize text.
        _word_vocab: dict. A mapping of words to feature indices.
        _char_vocab: dict. A mapping of chars to feature indices.
        _label_vocab: dict. A mapping of labels to feature indices.
    """

    def __init__(self, lower=True, num_norm=True,
                 use_char=True, initial_vocab=None):
        """Create a preprocessor object.

        Args:
            lower: boolean. Whether to convert the texts to lowercase.
            use_char: boolean. Whether to use char feature.
            num_norm: boolean. Whether to normalize text.
            initial_vocab: Iterable. Initial vocabulary for expanding word_vocab.
        """
        self._num_norm = num_norm
        self._use_char = use_char
        self._word_vocab = Vocabulary(lower=lower)
        self._char_vocab = Vocabulary(lower=False)
        self._label_vocab = Vocabulary(lower=False, unk_token=False)

        if initial_vocab:
            self._word_vocab.add_documents([initial_vocab])
            self._char_vocab.add_documents(initial_vocab)

    def fit(self, X, y):
        """Learn vocabulary from training set.

        Args:
            X : iterable. An iterable which yields either str, unicode or file objects.

        Returns:
            self : IndexTransformer.
        """
        self._word_vocab.add_documents(X)
        self._label_vocab.add_documents(y)
        if self._use_char:
            for doc in X:
                self._char_vocab.add_documents(doc)

        self._word_vocab.build()
        self._char_vocab.build()
        self._label_vocab.build()

        return self

    def transform(self, X, y=None):
        """Transform documents to document ids.

        Uses the vocabulary learned by fit.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.
            y : iterabl, label strings.

        Returns:
            features: document id matrix.
            y: label id matrix.
        """
        mentions = []
        mentions_char = []
        left_contexts = []
        right_contexts = []
        outputs = []

        word_ids = [self._word_vocab.doc2id(doc) for doc in X]
        char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X]
        ngram_indices = []
        for sent in word_ids:
            ngrams = self.generate_ngrams(sent, n=4)
            ngram_indices.append(ngrams)
            for l, r in ngrams:
                mentions.append(word_ids[l: r])
                mentions_char.append(char_ids[l:r])
                left_contexts.append(word_ids[:l])
                right_contexts.append(word_ids[r:])

        if y is not None:
            for ngram, labels in zip(ngram_indices, y):
                d = {(begin_offset, end_offset + 1): t for t, begin_offset, end_offset in get_entities(labels)}
                for l, r in ngram:
                    if (l, r) in d:
                        outputs.append(self._label_vocab[d[(l, r)]])
                    else:
                        outputs.append(self._label_vocab)

        outputs = np.array(outputs)
        inputs = [np.array(left_contexts), np.array(mentions), np.array(mentions_char), np.array(right_contexts)]

        if y is not None:
            return inputs, outputs
        else:
            return inputs

    def fit_transform(self, X, y=None, **params):
        """Learn vocabulary and return document id matrix.

        This is equivalent to fit followed by transform.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.

        Returns:
            list : document id matrix.
            list: label id matrix.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y, lengths=None):
        """Return label strings.

        Args:
            y: label id matrix.
            lengths: sentences length.

        Returns:
            list: list of list of strings.
        """
        y = np.argmax(y, -1)
        inverse_y = [self._label_vocab.id2doc(ids) for ids in y]
        if lengths is not None:
            inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]

        return inverse_y

    def generate_ngrams(self, words, n=7):
        res = []
        seqlen = len(words)
        for i in range(1, n + 1):
            for ngram in ngrams(range(seqlen), i):
                l, r = ngram[0], ngram[-1] + 1
                res.append((l, r))
        return res

    @property
    def word_vocab_size(self):
        return len(self._word_vocab)

    @property
    def char_vocab_size(self):
        return len(self._char_vocab)

    @property
    def label_size(self):
        return len(self._label_vocab)

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
