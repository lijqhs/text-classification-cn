
import os
import re
import jieba

token = "[0-9\s+\.\!\/_,$%^*()?;；：【】+\"\'\[\]\\]+|[+——！，;:。？《》、~@#￥%……&*（）“”.=-]+"

def preprocess(text):
    text1 = re.sub('&nbsp', ' ', text)
    str_no_punctuation = re.sub(token, ' ', text1)  # 去掉标点
    text_list = list(jieba.cut(str_no_punctuation))   # 分词列表
    text_list = [item for item in text_list if item != ' '] # 去掉空格
    return ' '.join(text_list)

def load_datasets():
    base_dir = 'data/'
    X_data = {'train':[], 'test':[]}
    y = {'train':[], 'test':[]}
    for type_name in ['train', 'test']:
        corpus_dir = os.path.join(base_dir, type_name)
        for label in os.listdir(corpus_dir):
            label_dir = os.path.join(corpus_dir, label)
            file_list = os.listdir(label_dir)
            print("label: {}, len: {}".format(label, len(file_list)))

            for fname in file_list:
                file_path = os.path.join(label_dir, fname)
                with open(file_path, encoding='gb2312', errors='ignore') as text_file:
                    text_content = preprocess(text_file.read())
                X_data[type_name].append(text_content)
                y[type_name].append(label)

        print("{} corpus len: {}\n".format(type_name, len(X_data[type_name])))
    
    return X_data['train'], y['train'], X_data['test'], y['test']
