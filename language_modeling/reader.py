"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import numpy as np


def _read_words(filename):
    with open(filename, 'r') as f:
        if sys.version_info[0] >= 3:
            return f.read().replace('\n', '<eos>').split()
        else:
            return f.read().decode('utf-8').replace('\n', '<eos>').split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The PTB dataset comes from Tomas Mikolov's webpage:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
        data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
        where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)

    return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, num_steps):
    """Create dataset for training.

    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        num_steps: int, the number of unrolls.

    Returns:
        X: input array shaped [num_data, num_steps].
        y: output array shaped [num_data, num_steps].
    """
    data_len = len(raw_data)
    num_data = (data_len - 1) // num_steps
    x = np.array([raw_data[i * num_steps: (i + 1) * num_steps] for i in range(num_data)])
    y = np.array([raw_data[i * num_steps + 1: (i + 1) * num_steps + 1] for i in range(num_data)])

    return x, y


if __name__ == '__main__':
    train_data, valid_data, test_data, vocabulary = ptb_raw_data(data_path='data/simple-examples/data')
    print(vocabulary)
    print(train_data[0:10])
    print(len(train_data))
    x, y = ptb_producer(train_data, num_steps=3)
    print(x[0])
    print(y[0])
    print(len(x))
    print(len(y))
