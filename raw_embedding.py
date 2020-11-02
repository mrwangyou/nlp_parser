import gensim

corpus = open('./dataset/corpus.txt', 'r', encoding='UTF-8')

for line in corpus:
    sent = line.split()
    for word in sent:
        if '[' in word:

