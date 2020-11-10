import re

# 去除词性标记/xxx，短语标记[]，标点；阿拉伯数字转汉语
file = open('2014_corpus.txt', encoding='UTF-8', errors='ignore')
lines = file.readlines()
lines = [re.sub('\[|]|\n', '', re.sub('/[a-zA-Z0-9]+', '', line)) for line in lines]
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
    # print(cnt)
    cnt = cnt+1
    for term in line.split():
        if 'a' <= term[0] <= 'z' or 'A' <= term[0] <= 'Z':
            line_tmp = line_tmp + term + ' '
        else:
            for word in term:
                if word != ' ':
                    if '\u4e00' <= word <= '\u9fa5' or 'a' <= word <= 'z' or 'A' <= word <= 'Z':
                        line_tmp = line_tmp + word + ' '
                    else:
                        if word in {'?', '？', '!', '！', '。', ',', '，', ';', '；', ':', '：', '《', '》', '“', '”', '\'',
                                    '%'}:
                            line_tmp = line_tmp + word + ' '
                            line_tmp = re.sub('\?|？', '问号', line_tmp)
                            line_tmp = re.sub('!|！', '感叹号', line_tmp)
                            line_tmp = re.sub('。', '句号', line_tmp)
                            line_tmp = re.sub(',|，', '逗号', line_tmp)
                            line_tmp = re.sub(';|；', '分号', line_tmp)
                            line_tmp = re.sub(':|：', '冒号', line_tmp)
                            line_tmp = re.sub('《|》', '书名号', line_tmp)
                            line_tmp = re.sub('“|”', '双引号', line_tmp)
                            line_tmp = re.sub('\'', '单引号', line_tmp)
                            line_tmp = re.sub('%', '百分号', line_tmp)
                        else:
                            line_tmp = line_tmp + '标标点点' + ' '
    tmp.append(line_tmp)
lines = tmp
file = open('2014_corpus_single_word.txt', 'w', encoding='UTF-8')
file.writelines([line + '\n' for line in lines])
file.close()
