# -*- coding: utf-8 -*-
"""
Character based language model.
"""
import re
from collections import Counter

import numpy as np
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def load_text(filename):
    """Load text into memory.

    Args:
        filename(str): text file.

    Returns:
        text(str): raw text.
    """
    with open(filename, 'r') as f:
        text = f.read()

    return text


def save_text(lines, filename):
    """Save text line by line.

    Args:
        lines(list): texts.
        filename(str): text file.
    """
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


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


def create_vocabulary(text):
    """Create vocabulary dictionaries.

    Args:
        text(str): raw text.

    Returns:
        char2id(dict): character to id mapping.
        id2char(dict): id to character mapping.
    """
    char2id = {'<PAD>': 0}
    id2char = {0: '<PAD>'}
    freq = Counter(text)
    for char, _ in freq.most_common():
        id = len(char2id)
        char2id[char] = id
        id2char[id] = char

    return char2id, id2char


def generate_text(model, char2id, id2char, seed_text, maxlen=10, iter=20):
    """Generate a sequence of characters.

    Args:
        model: trained model.
        char2id(dict): character to id mapping.
        id2char(dict): id to character mapping.
        maxlen: max sequence length.
        seed_text: seed text for generating new text.
        iter: number of iteration to generate character.

    Returns:
        text(str): generated text.
    """
    encoded = [char2id[char] for char in seed_text]
    for _ in range(iter):
        x = pad_sequences([encoded], maxlen=maxlen, truncating='pre')
        y = model.predict_classes(x, verbose=0)
        encoded.append(y[0])
    decoded = [id2char[c] for c in encoded]
    text = ''.join(decoded)

    return text


def clean_text(raw_text):
    """Clean raw text.

    Args:
        raw_text(str): text.

    Returns:
        cleaned_text(str): cleaned_text
    """
    tokens = raw_text.split()
    cleaned_text = ''.join(tokens)
    pattern = re.compile(r'（.+?）')
    cleaned_text = pattern.sub('', cleaned_text)

    return cleaned_text


def create_dataset(text, char2id, maxlen=10):
    """Create a dataset.

    Args:
        text(str): text.
        char2id(dict): character to id mapping.
        maxlen(int): max sequence length.

    Returns:
        X(ndarray): encoded character sequences.
        y(ndarray): encoded label sequences.
    """
    sequences = []
    for i in range(maxlen, len(text)):
        seq = text[i - maxlen: i + 1]
        encoded = [char2id[char] for char in seq]
        sequences.append(encoded)

    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]

    return X, y


def preprocess_dataset(X, y, vocab_size):
    X = [to_categorical(x, num_classes=vocab_size) for x in X]
    y = to_categorical(y, num_classes=vocab_size)

    return X, y


def main():
    raw_text = load_text('ja.text8')
    cleaned_text = clean_text(raw_text)
    cleaned_text = cleaned_text[:10000]
    print(cleaned_text)

    char2id, id2char = create_vocabulary(cleaned_text)
    vocab_size = len(char2id)
    print('Vocabulary size: {}'.format(vocab_size))

    X, y = create_dataset(cleaned_text, char2id)
    print('X shape: {}'.format(X.shape))
    print('y shape: {}'.format(y.shape))

    model = create_model(vocab_size)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=2)

    print(generate_text(model, char2id, id2char, 'ちょん掛けとは、相撲'))


if __name__ == '__main__':
    main()
