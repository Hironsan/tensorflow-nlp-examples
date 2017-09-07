import tensorflow as tf

from .crf import crf_decode


class NERModel(object):

    def __init__(self,
                 config,
                 embeddings,
                 word_ids,
                 sequence_lengths,
                 char_ids,
                 word_lengths,
                 labels,
                 dropout):
        self.config           = config
        self.embeddings       = embeddings
        self.word_ids         = word_ids
        self.sequence_lengths = sequence_lengths
        self.char_ids         = char_ids
        self.word_lengths     = word_lengths
        self.labels           = labels
        self.dropout          = dropout
        self.transition_params = tf.get_variable(name='transition',
                                                 shape=[self.config.num_tags, self.config.num_tags])

    def add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        with tf.variable_scope('words'):
            _word_embeddings = tf.Variable(self.embeddings,
                                           name='_word_embeddings',
                                           dtype=tf.float32,
                                           trainable=self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name='word_embeddings')

        with tf.variable_scope('chars'):
            if self.config.char_feature:
                # get embeddings matrix
                _char_embeddings = tf.get_variable(name='_char_embeddings',
                                                   dtype=tf.float32,
                                                   shape=[self.config.num_chars, self.config.char_embedding_size])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids,
                                                         name='char_embeddings')
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.config.char_embedding_size])
                word_lengths = tf.reshape(self.word_lengths, shape=[-1])
                # bi lstm on chars
                # need 2 instances of cells since tf 1.1
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.num_char_lstm_units, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.num_char_lstm_units, state_is_tuple=True)

                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                      cell_bw,
                                                                                      char_embeddings,
                                                                                      sequence_length=word_lengths,
                                                                                      dtype=tf.float32)

                output = tf.concat([output_fw, output_bw], axis=-1)
                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[-1, s[1], 2 * self.config.num_char_lstm_units])

                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope('bi-lstm'):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.num_word_lstm_units)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.num_word_lstm_units)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                        cell_bw,
                                                                        self.word_embeddings,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope('proj'):
            W = tf.get_variable('W', shape=[2 * self.config.num_word_lstm_units, self.config.num_tags], dtype=tf.float32)

            b = tf.get_variable('b', shape=[self.config.num_tags], dtype=tf.float32, initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.num_word_lstm_units])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.config.num_tags])

            if self.config.crf:
                self.output, _ = crf_decode(self.logits, self.transition_params, self.sequence_lengths)
            else:
                self.output = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        """
        Adds loss to self
        """
        if self.config.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar('loss', self.loss)

    def build(self):
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()
