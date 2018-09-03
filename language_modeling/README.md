# Language Modeling

In this tutorial, we will show how to train some neural networks on a challenging task of language modeling. The goal of the problem is to fit a probabilistic model which assigns probabilities to sentences. It does so by predicting next words $w_i$ in a text given a history of previous words $w_1,w_2,\ldots,w_{i-1}$. For training the model, we will use the Penn Tree Bank (PTB) dataset, which is a popular benchmark for measuring the quality of these models, whilst being small and relatively fast to train.

Language modeling is key to many interesting problems such as speech recognition, machine translation, or image captioning. It is also fun -- take a look here.


## Tutorial Files

This tutorial references the following files:

File         | What's in it?
------------ | -------------
`data/`       | PTB dataset.
`char_lstm.py` | The code to train a character based LSTM language model on the PTB dataset.
`word_lstm.py` | The code to train a word based LSTM language model.

## Download and Prepare the Data

The data required for this tutorial is in the data/ directory of the [PTB dataset from Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz).

```bash
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvfz simple-examples.tgz -C data/
$ rm simple-examples.tgz
```

The dataset is already preprocessed and contains overall 10000 different words, including the end-of-sentence marker and a special symbol (\<unk\>) for rare words. In reader.py, we convert each word to a unique integer identifier, in order to make it easy for the neural network to process the data.

## Run the Code

To train a language model, run the following code:

```bash
$ python char_lstm.py --data_path=data/simple-examples/data
```

## License

[MIT](https://github.com/Hironsan/tensorflow-nlp-examples/blob/master/LICENSE)