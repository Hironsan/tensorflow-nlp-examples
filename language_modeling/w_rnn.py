import random
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


def RNN(inputs, weights, biases, n_hidden):
    # cell = tf.contrib.rnn.MultiRNNCell([cell] * 2) # why does not this work?
    # cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(2)]) # OK
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden), tf.contrib.rnn.BasicLSTMCell(n_hidden)])
    outputs, states = tf.contrib.rnn.static_rnn(cell, [inputs], dtype=tf.float32)
    # outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases


def create_rnn(seq, n_hidden):
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(n_hidden) for _ in range(2)])
    outputs, states = tf.contrib.rnn.static_rnn(cell, [seq], dtype=tf.float32)
    return outputs, states

def create_model(seq, temp, n_hidden):
    outputs, states = create_rnn(seq, n_hidden)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, None)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=temp))
    return loss, states


def training(seq, temp, loss, states, optimizer):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        offset = random.randint(0, n_input + 1)
        end_offset = n_input + 1
        loss_total = 0
        for step in range(training_iters):
            if offset > (len(words) - end_offset):
                offset = random.randint(0, n_input + 1)

            symbols_in_keys = [[dictionary[words[i]] for i in range(offset, offset + n_input)]]

            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[words[offset + n_input]]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

            _, batch_loss = sess.run([optimizer, loss], feed_dict={seq: symbols_in_keys, temp:symbols_out_onehot})
            loss_total += batch_loss
            if (step + 1) %  display_step == 0:
                print('Iter= {}, Average Loss= {:.6f}'.format(step + 1, loss_total / display_step))
                loss_total = 0
            offset += n_input + 1


def main():
    x = tf.placeholder(shape=[None, n_input], dtype=tf.float32)
    y = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32)
    loss, states = create_model(x, y, n_hidden)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    training(x, y, loss, states, optimizer)


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

    main()
    """
    # Graph input
    x = tf.placeholder(shape=[None, n_input], dtype=tf.float32)
    y = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32)

    weights = tf.Variable(initial_value=tf.random_normal(shape=[n_hidden, vocab_size]), name='weights')
    biases = tf.Variable(initial_value=tf.random_normal(shape=[vocab_size]), name='biases')

    pred = RNN(x, weights, biases, n_hidden)
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    loss = tf.reduce_mean(entropy)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        offset = random.randint(0, n_input + 1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0

        for step in range(training_iters):
            if offset > (len(words) - end_offset):
                offset = random.randint(0, n_input + 1)

            symbols_in_keys = [[dictionary[words[i]] for i in range(offset, offset + n_input)]]

            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[words[offset + n_input]]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

            _, batch_acc, batch_loss, onehot_pred = sess.run([optimizer, accuracy, loss, pred],
                                                 feed_dict={x: symbols_in_keys, y:symbols_out_onehot})
            loss_total += batch_loss
            acc_total += batch_acc
            if (step + 1) %  display_step == 0:
                print(step+1, loss_total/display_step, 100*acc_total/display_step)
                acc_total = 0
                loss_total = 0
                symbols_in = [words[i] for i in range(offset, offset + n_input)]  # 入力単語n_input数分
                symbols_out = words[offset + n_input]  # 出力単語
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]  # 実際に予測した単語
                print(symbols_in, symbols_out, symbols_out_pred)
            offset += n_input + 1
    """