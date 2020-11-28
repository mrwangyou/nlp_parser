import torch
import re
import numpy
import joblib

if __name__ == '__main__':
    mydict = {}
    f = open('./dataset/2014_corpus_single_word_dict.txt', 'r', encoding='UTF-8')
    for line in f:
        key, char = line.split()
        mydict[char] = key
    f.close()

    f = open('./dataset/2014_corpus.txt', 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    lines = [re.sub('\n', '', line) for line in lines]
    lines = [re.sub('\[/w', ' ', line) for line in lines]
    lines = [re.sub(']/w', ' ', line) for line in lines]
    lines = [re.sub('\[', ' [ ', line) for line in lines]
    lines = [re.sub(']', ' ] ', line) for line in lines]
    lines = [re.sub('/[a-zA-Z0-9]+ ', ' ', line) for line in lines]
    lines = [re.sub('[^0-9\u4e00-\u9fa5\[\] ]', ' ', line) for line in lines]
    lines = [re.sub('[ ]+', ' ', line) for line in lines]

    tmp = []
    for line in lines:
        line_tmp = line
        line_tmp = re.sub('0', '零', line_tmp)
        line_tmp = re.sub('1', '一', line_tmp)
        line_tmp = re.sub('2', '二', line_tmp)
        line_tmp = re.sub('3', '三', line_tmp)
        line_tmp = re.sub('4', '四', line_tmp)
        line_tmp = re.sub('5', '五', line_tmp)
        line_tmp = re.sub('6', '六', line_tmp)
        line_tmp = re.sub('7', '七', line_tmp)
        line_tmp = re.sub('8', '八', line_tmp)
        line_tmp = re.sub('9', '九', line_tmp)
        line_tmp = re.sub('(?<=[a-zA-Z])(?=[\u4e00-\u9fa5])', ' ', line_tmp)
        line_tmp = re.sub('(?<=[\u4e00-\u9fa5])(?=[a-zA-Z])', ' ', line_tmp)
        tmp.append(line_tmp)
    lines = tmp

    # 词形tensor
    # 非短语内容且词语开头    非短语内容且词语中间  非短语内容且词语结尾  非短语内容且单词语[0,1,2,3]   短语[4,5,6,7]
    num = 0
    flag = 0
    dataset = []
    for line in lines:
        sentSet = []
        num = num + 1
        l = line.split()
        term_i = 0
        while term_i < len(l):
            term = l[term_i]
            if (term != '[') or (term == '[' and ']' not in l[term_i:]):
                if term != '[':
                    if 'a' <= term[0] <= 'z' or 'A' <= term[0] <= 'Z':
                        sentSet.append(3)
                    else:
                        if len(term) == 1:
                            sentSet.append(3)
                        else:
                            sentSet.append(0)
                            for i in range(len(term) - 2):
                                sentSet.append(1)
                            sentSet.append(2)
            else:
                term_i = term_i + 1
                while l[term_i] != ']':
                    term = l[term_i]
                    if 'a' <= term[0] <= 'z' or 'A' <= term[0] <= 'Z':
                        sentSet.append(7)
                    else:
                        if len(term) == 1:
                            sentSet.append(7)
                        else:
                            sentSet.append(4)
                            for i in range(len(term) - 2):
                                sentSet.append(5)
                            sentSet.append(6)
                    term_i = term_i + 1
            term_i = term_i + 1

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
                    tmp.append(8)
                finalSet.append(tmp)
            # print(torch.tensor(finalSet))
            torch.save(torch.tensor(finalSet), './dataset/word_dict_tensor/' + str(flag) + 'p.pth')
            flag = flag + 1
            dataset = []
            print(flag)

    finalSet = []
    maxSize = 0
    for line in dataset:
        maxSize = max(maxSize, len(line))
    for line in dataset:
        tmp = line
        while len(tmp) < maxSize:
            tmp.append(8)
        finalSet.append(tmp)
    torch.save(torch.tensor(finalSet), './dataset/word_dict_tensor/' + str(flag) + 'p.pth')

    ten_dict = {}
    for i in range(flag + 1):
        x = torch.load('./dataset/word_dict_tensor/' + str(i) + '.pth')
        y = torch.load('./dataset/word_dict_tensor/' + str(i) + 'p.pth')
        ten_dict[x] = y
    joblib.dump(ten_dict, 'tsr_dict.pkl')
