"""
Word based language model.
"""
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential

from .reader import ptb_raw_data, ptb_producer


def create_model(vocab_size, embedding_dim=50):
    """Create a model.

    Args:
        vocab_size(int): vocabulary size.
        embedding_dim(int): embedding dimension.

    Returns:
        model: model object.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True))
    model.add(LSTM(75))
    model.add(Dense(vocab_size, activation='softmax'))

    return model


def main():
    train_data, valid_data, test_data, vocab_size = ptb_raw_data(data_path='data/simple-examples/data')
    X, y = ptb_producer(train_data, num_steps=3)
    print('Vocabulary size: {}'.format(vocab_size))

    print('X shape: {}'.format(X.shape))
    print('y shape: {}'.format(y.shape))

    model = create_model(vocab_size)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=2)


if __name__ == '__main__':
    main()
