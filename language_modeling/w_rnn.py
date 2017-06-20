from collections import Counter

import numpy as np
import tensorflow as tf


def load_file(file_name):
    with open(file_name) as f:
        line = f.readline()
    words = np.array(line.strip().split())
    return words


def build_vocabulary(words):
    count = Counter(words).most_common()
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = {v: k for k, v in dictionary.items()}
    return dictionary, reverse_dictionary


def RNN(x, weights, biases, n_hidden):
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases


if __name__ == '__main__':
    training_file = 'belling_the_cat.txt'
    words = load_file(training_file)
    dictionary, reverse_dictionary = build_vocabulary(words)
    vocab_size = len(dictionary)

    # Hyper parameters
    learning_rate = 0.001
    training_iters = 50000
    display_step = 1000
    n_input = 3
    n_hidden = 512

    # Graph input
    x = tf.placeholder(shape=[None, n_input, 1], dtype=tf.float32)
    y = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32)

    weights = tf.Variable(initial_value=tf.random_normal(shape=[n_hidden, vocab_size]), name='weights')
    biases = tf.Variable(initial_value=tf.random_normal(shape=[vocab_size]), name='biases')

    pred = RNN(x, weights, biases, n_hidden)
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    loss = tf.reduce_mean(entropy)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)