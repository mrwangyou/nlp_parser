import re

# 去除词性标记/xxx，短语标记[]，标点；阿拉伯数字转汉语
file = open('./dataset/2014_corpus.txt', 'r', encoding='UTF-8', errors='ignore')
lines = file.readlines()
file.close()
lines = [re.sub('\[|]|\n', ' ', line) for line in lines]
lines = [re.sub('/[a-zA-Z0-9]+ ', ' ', line) for line in lines]
lines = [re.sub('[^0-9\u4e00-\u9fa5 ]', ' ', line) for line in lines]
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

# 添加空格
tmp = []
cnt = 0
for line in lines:
    line_tmp = ''
    print(cnt)
    cnt = cnt + 1
    for term in line.split():
        if 'a' <= term[0] <= 'z' or 'A' <= term[0] <= 'Z':
            line_tmp = line_tmp + term + ' '
        else:
            for word in term:
                line_tmp = line_tmp + word + ' '
    tmp.append(line_tmp)
lines = tmp
file = open('./dataset/2014_corpus_single_word.txt', 'w', encoding='UTF-8')
file.writelines([line + '\n' for line in lines])
file.close()
