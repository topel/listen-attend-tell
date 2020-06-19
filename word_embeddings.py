
import numpy as np
# import os
# from random import shuffle
import random

from gensim.models import Word2Vec, FastText
import pickle

np.random.seed(0)
random.seed(0)

import torch

import socket
host_name = socket.gethostname()
print(host_name)

__author__ = "Thomas Pellegrini - 2020"

data_dir = '../clotho-dataset/data'

WORD_LIST = pickle.load(open(data_dir + "/words_list.p", "rb"))# 4367 word types

def load_text_into_a_list_of_sentences(fpath, sentences=None):
    if sentences is None: sentences = []
    with open(fpath, "rt") as fh:
        for ligne in fh:
            sentences.append(ligne.rstrip().replace('<sos>', '').replace('<eos>', '').split())
    return sentences

input_text='../clotho-dataset/lm/dev.txt'

sentences_dev = load_text_into_a_list_of_sentences(input_text)
input_text='../clotho-dataset/lm/eva.txt'
# 14465 phrases

sentences_dev = load_text_into_a_list_of_sentences(input_text, sentences_dev)
# 19690 phrases

print(sentences_dev[:2])
print(len(sentences_dev))


emb_dim=128

# model = FastText(size=emb_dim, window=3, min_count=1)
model = Word2Vec(size=emb_dim, window=3, min_count=1)
model.build_vocab(sentences=sentences_dev)
model.train(sentences=sentences_dev, total_examples=len(sentences_dev), epochs=10)
# Word2Vec()
# model = Word2Vec(sentences=sentences_dev, size=128, window=3, min_count=1, workers=4, sg=0, iter=10)
print(model.wv.most_similar("man"))
# print(model.wv.most_similar("<sos>"))
# print(model.wv.most_similar("<eos>"))

dev_embeddings = []
for i,w in enumerate(WORD_LIST):
    if w == '<sos>' or w == '<eos>':
        # dev_embeddings.append(np.zeros(emb_dim,))
        dev_embeddings.append(np.random.normal(scale=0.6, size=(emb_dim,)))
        continue
    dev_embeddings.append(model.wv[w])

# print(WORD_LIST[0], dev_embeddings[0].shape, dev_embeddings[0])
# print(WORD_LIST[1], dev_embeddings[1].shape, dev_embeddings[1])
print(len(dev_embeddings))

dev_embeddings = torch.FloatTensor(dev_embeddings)
print(dev_embeddings.size())

# torch.save(dev_embeddings, "../clotho-dataset/lm/word2vec_dev_128.pth")
torch.save(dev_embeddings, "../clotho-dataset/lm/word2vec_dev_eva_128.pth")
# torch.save(dev_embeddings, "../clotho-dataset/lm/fasttext_dev_128.pth")
# torch.save(dev_embeddings, "../clotho-dataset/lm/fasttext_dev_eva_128.pth")
