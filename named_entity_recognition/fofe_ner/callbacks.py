"""
Custom callbacks.
"""
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, classification_report


class F1score(Callback):

    def __init__(self, seq, preprocessor=None):
        super(F1score, self).__init__()
        self.seq = seq
        self.p = preprocessor

    def on_epoch_end(self, epoch, logs={}):
        label_true = []
        label_pred = []
        for i in range(len(self.seq)):
            x_true, y_true = self.seq[i]
            y_true = np.argmax(y_true, -1)

            y_pred = self.model.predict_on_batch(x_true)
            y_pred = np.argmax(y_pred, -1)

            label_true.extend(y_true)
            label_pred.extend(y_pred)

        f1_micro = f1_score(label_true, label_pred, average='micro')
        f1_macro = f1_score(label_true, label_pred, average='macro')
        print(' - f1(micro): {:04.2f}'.format(f1_micro * 100))
        print(' - f1(macro): {:04.2f}'.format(f1_macro * 100))
        print(classification_report(label_true, label_pred))
        logs['f1'] = f1_micro
