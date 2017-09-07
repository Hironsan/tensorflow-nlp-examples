import tensorflow as tf
import tqdm

from .eval import f1_score


class Trainer(object):

    def __init__(self,
                 model,
                 model_config,
                 training_config,
                 checkpoint_path='',
                 save_path='',
                 tensorboard=True,
                 embeddings=None,
                 preprocessor=None
                 ):

        self.model_config = model_config
        self.training_config = training_config
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.p = preprocessor

        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='char_ids')
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name='word_lengths')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

        self.model = model(model_config,
                           embeddings,
                           self.word_ids,
                           self.sequence_lengths,
                           self.char_ids,
                           self.word_lengths,
                           self.labels,
                           self.dropout)
        self.model.build()

    def train(self, train_steps, train_batches, valid_steps=None, valid_batches=None):
        best_score = 0.0
        saver = tf.train.Saver()
        train_op = self.get_train_op()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.add_summary(sess)
            for epoch in range(self.training_config.max_epoch):
                print('Epoch {}/{}'.format(epoch + 1, self.training_config.max_epoch))
                self.training_config.learning_rate *= self.training_config.lr_decay
                self.run_epoch(train_steps, train_batches, train_op, epoch, sess)
                f1 = self.validate(valid_steps, valid_batches, sess)
                self.save(sess, saver, best_score, f1)
                best_score = f1 if f1 >= best_score else best_score

    def save(self, sess, saver, best_score, current_score):
        if current_score >= best_score:
            print(self.save_path)
            saver.save(sess, self.save_path + self.training_config.model_name)

    def run_epoch(self, train_steps, train_batches, train_op, epoch, sess):
        for i in tqdm.tqdm(range(train_steps)):
            data, labels = next(train_batches)
            fd, _ = self.get_feed_dict(data, labels,
                                       self.training_config.learning_rate,
                                       self.training_config.dropout)
            _, train_loss, summary = sess.run([train_op, self.model.loss, self.merged], feed_dict=fd)

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * train_steps + i)

    def validate(self, valid_steps, valid_batches, sess):
        y_trues, y_preds = [], []
        seq_lengths = []
        for i in range(valid_steps):
            data, y_true = next(valid_batches)
            fd, sequence_lengths = self.get_feed_dict(data, dropout=1.0)
            y_pred = sess.run([self.model.output], feed_dict=fd)
            y_pred = y_pred[0]
            y_true = [self.p.inverse_transform(y[:l]) for y, l in zip(y_true, sequence_lengths)]
            y_pred = [self.p.inverse_transform(y[:l]) for y, l in zip(y_pred, sequence_lengths)]
            y_trues.extend(y_true)
            y_preds.extend(y_pred)
            seq_lengths.extend(sequence_lengths)
        f1 = f1_score(y_trues, y_preds, seq_lengths)
        print(' - f1: {:04.2f}'.format(f1 * 100))

        return f1

    def get_feed_dict(self, data, labels=None, lr=None, dropout=None):
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

        return feed, sequence_lengths

    def get_train_op(self):
        """Gets train_op.
        """
        with tf.variable_scope('train_step'):
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

            # gradient clipping
            if self.training_config.clip_value > 0:
                gradients, variables = zip(*optimizer.compute_gradients(self.model.loss))
                gradients, global_norm = tf.clip_by_global_norm(gradients, self.training_config.clip_value)
                train_op = optimizer.apply_gradients(zip(gradients, variables))
            else:
                train_op = optimizer.minimize(self.model.loss)

            return train_op

    def add_summary(self, sess):
        # for tensorboard
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.checkpoint_path, sess.graph)
