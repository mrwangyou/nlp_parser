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

f = open('./dataset/2014_corpus_single_word_dict.txt', 'r', encoding='UTF-8')
for line in f:
    key, char = line.split()
    dict[char] = key
f.close()

f = open('./dataset/2014_corpus_single_word.txt', 'r', encoding='UTF-8')
num = 0
flag = 0
dataset = []
for sent in f.readlines():
    sentSet = []
    num = num + 1
    for char in sent.split():
        sentSet.append(int(dict[char]))
    dataset.append(sentSet)
    if num >= 100:
        num = 0
        finalSet = []
        maxSize = 0
        for line in dataset:
            maxSize = max(maxSize, len(line))
        for line in dataset:
            tmp = line
            while len(tmp) < maxSize:
                tmp.append(len(dict))
            finalSet.append(tmp)
        # print(torch.tensor(finalSet))
        torch.save(torch.tensor(finalSet), './dataset/word_dict_tensor/' + str(flag) + '.pth')
        flag = flag + 1
        dataset = []
finalSet = []
maxSize = 0
for line in dataset:
    maxSize = max(maxSize, len(line))
for line in dataset:
    tmp = line
    while len(tmp) < maxSize:
        tmp.append(len(dict))
    finalSet.append(tmp)
torch.save(torch.tensor(finalSet), './dataset/word_dict_tensor/' + str(flag) + '.pth')
