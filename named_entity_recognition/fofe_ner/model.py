"""
Model definition.
"""
from keras.layers import Dense, LSTM, Embedding, Input, Dropout
from keras.layers import Bidirectional, BatchNormalization, TimeDistributed
from keras.models import Model
from keras.layers.merge import Concatenate


class CharModel(object):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Mingbin Xu, Hui Jiang, Sedtawut Watcharawittayakul
    A Local Detection Approach for Named Entity Recognition and Mention Detection. ACL 2017.
    http://aclweb.org/anthology/P17-1114
    """

    def __init__(self, word_vocab_size, char_vocab_size, num_labels,
                 word_emb_size=100, word_lstm_units=50,
                 char_emb_size=25, char_lstm_units=25,
                 dropout=0.5, embeddings=None):
        """Build a FOFE NER model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_emb_size (int): word embedding dimensions.
            char_emb_size (int): character embedding dimensions.
            word_lstm_units (int): character LSTM feature extractor output dimensions.
            char_lstm_units (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
            use_char (boolean): add char feature.
        """
        self._word_emb_size = word_emb_size
        self._word_lstm_units = word_lstm_units
        self._word_vocab_size = word_vocab_size
        self._char_emb_size = char_emb_size
        self._char_lstm_units = char_lstm_units
        self._char_vocab_size = char_vocab_size
        self._dropout = dropout
        self._embeddings = embeddings
        self._num_labels = num_labels

    def build(self):
        left_context = Input(batch_shape=(None, None), dtype='int32')
        mention = Input(batch_shape=(None, None), dtype='int32')
        mention_char = Input(batch_shape=(None, None, None), dtype='int32')
        right_context = Input(batch_shape=(None, None), dtype='int32')

        embeddings = Embedding(input_dim=self._embeddings.shape[0],
                               output_dim=self._embeddings.shape[1],
                               mask_zero=True,
                               weights=[self._embeddings])
        left_embeddings = embeddings(left_context)
        mention_embeddings = embeddings(mention)
        right_embeddings = embeddings(right_context)
        char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                    output_dim=self._char_emb_size,
                                    mask_zero=True
                                    )(mention_char)

        char_embeddings = TimeDistributed(Bidirectional(LSTM(self._char_lstm_units)))(char_embeddings)
        mention_embeddings = Concatenate(axis=-1)([mention_embeddings, char_embeddings])

        x1 = Bidirectional(LSTM(units=self._word_lstm_units))(left_embeddings)
        x2 = Bidirectional(LSTM(units=self._word_lstm_units))(mention_embeddings)
        x3 = Bidirectional(LSTM(units=self._word_lstm_units))(right_embeddings)

        x = Concatenate()([x1, x2, x3])
        x = BatchNormalization()(x)
        x = Dense(self._word_lstm_units, activation='tanh')(x)
        pred = Dense(self._num_labels, activation='softmax')(x)

        model = Model(inputs=[left_context, mention, mention_char, right_context], outputs=[pred])

        return model
