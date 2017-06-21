import random
from collections import Counter

import numpy as np
import tensorflow as tf

DATA_PATH = 'belling_the_cat.txt'
LEARNING_RATE = 0.001
N_EPOCKS = 50000
SKIP_STEP = 1000
NUM_STEPS = 3
HIDDEN_SIZE = 512


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


def create_rnn(seq, n_hidden):
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(n_hidden) for _ in range(2)])
    outputs, states = tf.contrib.rnn.static_rnn(cell, [seq], dtype=tf.float32)
    return outputs, states

def create_model(seq, temp, vocab_size, n_hidden=HIDDEN_SIZE):
    outputs, states = create_rnn(seq, n_hidden)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, None)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=temp))
    return loss, states


def training(seq, temp, loss, states, optimizer, words, dictionary, reverse_dictionary, vocab_size):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        offset = random.randint(0, NUM_STEPS + 1)
        end_offset = NUM_STEPS + 1
        loss_total = 0
        for step in range(N_EPOCKS):
            if offset > (len(words) - end_offset):
                offset = random.randint(0, NUM_STEPS + 1)

            symbols_in_keys = [[dictionary[words[i]] for i in range(offset, offset + NUM_STEPS)]]

            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[words[offset + NUM_STEPS]]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

            _, batch_loss = sess.run([optimizer, loss], feed_dict={seq: symbols_in_keys, temp:symbols_out_onehot})
            loss_total += batch_loss
            if (step + 1) %  SKIP_STEP == 0:
                print('Iter= {}, Average Loss= {:.6f}'.format(step + 1, loss_total / SKIP_STEP))
                loss_total = 0
            offset += NUM_STEPS + 1


def main():
    words = load_file(DATA_PATH)
    dictionary, reverse_dictionary = build_vocabulary(words)
    vocab_size = len(dictionary)
    x = tf.placeholder(shape=[None, NUM_STEPS], dtype=tf.float32)
    y = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32)
    loss, states = create_model(x, y, vocab_size)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    training(x, y, loss, states, optimizer, words, dictionary, reverse_dictionary, vocab_size)


if __name__ == '__main__':
    main()
