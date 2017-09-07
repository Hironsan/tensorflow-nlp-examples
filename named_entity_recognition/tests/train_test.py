import os
import unittest

from ..config import ModelConfig, TrainingConfig
from ..train import Trainer
from ..data_loader import load_data_and_labels, load_word_embeddings, batch_iter
from ..preprocess import prepare_preprocessor
from ..model import NERModel


class TestTrainer(unittest.TestCase):

    def test_train(self):
        DATA_ROOT = os.path.join(os.path.dirname(__file__), '../data/ner')
        SAVE_ROOT = os.path.join(os.path.dirname(__file__), '../models')  # trained model
        LOG_ROOT = os.path.join(os.path.dirname(__file__), '../logs')     # checkpoint, tensorboard
        embedding_path = os.path.join(os.path.dirname(__file__), '../data/glove.6B/glove.6B.100d.txt')

        model_config = ModelConfig()
        training_config = TrainingConfig()

        train_path = os.path.join(DATA_ROOT, 'train.txt')
        valid_path = os.path.join(DATA_ROOT, 'valid.txt')
        x_train, y_train = load_data_and_labels(train_path)
        x_valid, y_valid = load_data_and_labels(valid_path)

        p = prepare_preprocessor(x_train, y_train)
        model_config.num_words = len(p.vocab_word)
        model_config.num_chars = len(p.vocab_char)
        model_config.num_tags  = len(p.vocab_tag)
        embeddings = load_word_embeddings(p.vocab_word, embedding_path, model_config.word_embedding_size)

        train_steps, train_batches = batch_iter(x_train, y_train, training_config.batch_size, preprocessor=p)
        valid_steps, valid_batches = batch_iter(x_valid, y_valid, training_config.batch_size, preprocessor=p)

        trainer = Trainer(NERModel,
                          model_config,
                          training_config,
                          checkpoint_path=LOG_ROOT,
                          save_path=SAVE_ROOT,
                          embeddings=embeddings,
                          preprocessor=p)
        trainer.train(train_steps, train_batches, valid_steps, valid_batches)

        p.save(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
