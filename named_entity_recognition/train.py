import tensorflow as tf


class Trainer(object):

    def __init__(self,
                 model,
                 model_config,
                 training_config,
                 checkpoint_path='',
                 save_path='',
                 tensorboard=True,
                 embeddings=None
                 ):

        self.model_config = model_config
        self.training_config = training_config
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.tensorboard = tensorboard

        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='char_ids')
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name='word_lengths')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        self.train_phase = tf.placeholder(tf.bool, [])

        self.model = model(model_config,
                           embeddings,
                           self.word_ids,
                           self.sequence_lengths,
                           self.char_ids,
                           self.word_lengths,
                           self.labels,
                           self.dropout,
                           self.train_phase)
        self.model.build()

    def train(self, train_steps, train_batches, valid_steps=None, valid_batches=None):
        train_op = self.get_train_op()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.training_config.max_epoch):
                self.training_config.learning_rate *= self.training_config.lr_decay
                self.run_epoch(train_steps, train_batches, train_op, sess)
                self.validate(valid_steps, valid_batches, sess)

    def run_epoch(self, train_steps, train_batches, train_op, sess):
        for i in range(train_steps):
            data, labels = next(train_batches)
            fd, _ = self.get_feed_dict(data, labels,
                                       self.training_config.learning_rate,
                                       self.training_config.dropout,
                                       train_phase=True)
            _, train_loss = sess.run([train_op, self.model.loss], feed_dict=fd)

    def validate(self, valid_steps, valid_batches, sess):
        for i in range(valid_steps):
            data, labels = next(valid_batches)
            fd, sequence_lengths = self.get_feed_dict(data, dropout=1.0)
            output = sess.run([self.model.output], feed_dict=fd)

    def get_feed_dict(self, data, labels=None, lr=None, dropout=None, train_phase=False):
        """
        Builds a feed dictionary.
        """
        feed = {}

        if self.model_config.char_feature:
            word_ids, char_ids, sequence_lengths, word_lengths = data
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths
        else:
            word_ids, sequence_lengths = data

        feed[self.word_ids] = word_ids
        feed[self.sequence_lengths] = sequence_lengths

        if labels is not None:
            feed[self.labels] = labels

        if lr:
            feed[self.lr] = lr

        if dropout:
            feed[self.dropout] = dropout

        feed[self.train_phase] = train_phase

        return feed, sequence_lengths

    def get_train_op(self):
        """Gets train_op.
        """
        with tf.variable_scope('train_step'):
            # sgd method
            if self.training_config.lr_method == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.training_config.lr_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif self.training_config.lr_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif self.training_config.lr_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise NotImplementedError('Unknown train op {}'.format(self.training_config.lr_method))

            # gradient clipping if config.clip_value is positive
            if self.training_config.clip_value > 0:
                gradients, variables = zip(*optimizer.compute_gradients(self.model.loss))
                gradients, global_norm = tf.clip_by_global_norm(gradients, self.training_config.clip_value)
                train_op = optimizer.apply_gradients(zip(gradients, variables))
            else:
                train_op = optimizer.minimize(self.model.loss)

            return train_op

    def add_summary(self):
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.checkpoint_path, self.sess.graph)
