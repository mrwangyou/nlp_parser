# -*- coding:utf-8 -*-
#
# for i, word in enumerate(wv.vocab):
#     if i == 10:
#         break
#     print(word)
# print(wv['king'])

# np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

from gensim.test.utils import datapath
from gensim import utils
import time
import numpy as np
import gensim.models
import pandas as pd
from scipy.stats import spearmanr
import re

path = r"C:\Users\mrwan\Desktop"

tBegin = time.time()


class MyCorpus(object):
    def __iter__(self):
        corpus_path = datapath('index.txt')
        for line in open(corpus_path, encoding='UTF-8', errors='ignore'):
            tmp = ''
            for word in line:
                tmp = tmp + word + word
            yield utils.simple_preprocess(tmp)


sentences = MyCorpus()

model = gensim.models.Word2Vec(
    sentences=sentences,
    min_count=0,
    size=300,
)

# for word in model.wv.vocab:
#     print(word)

# vec_to = model.wv['ball']
# print(vec_to)
f = open('./embedding.txt', 'w', encoding='UTF-8')
for _, word in enumerate(model.wv.vocab):
    f.write(word)
    for i in range(300):
        f.write(' ' + model.wv[word][i])
    f.write('\n')
f.close()

