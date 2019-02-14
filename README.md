# QuoraQuest

This project implements a two models deciding if two questions share the same intent.

### Prerequisites
In order to train a model from scrach you need to have the following:
- python 3.6
- torch
- numpy
- pandas
- skylearn

In addition, you need fasttext word embeddings file. get it [here](https://fasttext.cc/docs/en/english-vectors.html).
- Download wiki-news-300d-1M.vec.zip
- Unzip it and name the file fasttext.vec
- locate fasttext.vec in Project/Data/

### Training the simple model
```shell
$ python3.6 main.py --model simple --lr 0.0001 --batch_size 128 -- epochs 250 --test_ratio 0.1 --cuda True
```

### Training the simple model
```shell
$ python3.6 main.py --model attn --lr 0.0001 --batch_size 128 -- epochs 250 --test_ratio 0.1 --cuda True
```
You can control and change the hyper-parameters to try different settings.

In order to run without a gpu set the cuda parameter to False

Our dataset is taken from [here](https://www.kaggle.com/c/quora-question-pairs/data).

