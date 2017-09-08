# Named-Entity Recognition with Tensorflow
This repository implements a sequence tagging model using TensorFlow (BiLSTM + CRF + char embeddings).

## Data and Word Vectors
The data must be in the following format(tsv).
We provide an example in train.txt:

```
EU	B-ORG
rejects	O
German	B-MISC
call	O
to	O
boycott	O
British	B-MISC
lamb	O
.	O

Peter	B-PER
Blackburn	I-PER
```

You also need to download [GloVe vectors](https://nlp.stanford.edu/projects/glove/) and store it in *data/glove.6B* directory.


## Getting Started
I have already implemented training code.
 You have only to perform it.

```commandline
$ python -m unittest tests.train_test
```