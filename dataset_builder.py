import torch

dict = {}

f = open('./dataset/2014_corpus_single_word.txt', 'r', encoding='UTF-8')
for sent in f:
    for char in sent.split():
        dict[char] = 1
f.close()

f = open('./dataset/2014_corpus_single_word_dict.txt', 'w', encoding='UTF-8')
for key, char in enumerate(dict):
    f.write(str(key) + ' ' + char + '\n')
f.close()

f = open('./dataset/2014_corpus_single_word_dict.txt', 'w', encoding='UTF-8')
for line in f:
    key, char = line.split()
    dict[char] = key
f.close()

f = open('./dataset/2014_corpus_single_word.txt', 'r', encoding='UTF-8')
num = 0
dataset = []
for sent in f:

    num = num + 1
    for char in sent.split():
