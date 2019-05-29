
import os
import re
import jieba
import time
import numpy as np

jieba.enable_parallel() # jieba支持多进程

token = "[0-9\s+\.\!\/_,$%^*()?;；：【】+\"\'\[\]\\]+|[+——！，;:。？《》、~@#￥%……&*（）“”.=-]+"
labels_index = {} # 记录分类标签的序号
stopwords = set(open('dict/stop_words.txt', encoding='utf-8').read().split()) # 停用词


# for scikit part

def preprocess(text):
    text1 = re.sub('&nbsp', ' ', text)
    str_no_punctuation = re.sub(token, ' ', text1)  # 去掉标点
    text_list = list(jieba.cut(str_no_punctuation))   # 分词列表
    text_list = [item for item in text_list if item != ' '] # 去掉空格
    return ' '.join(text_list)


def load_datasets():
    # should run corpus_split.py first

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


# for keras part

def preprocess_keras(text):
    text1 = re.sub('&nbsp', ' ', text)
    str_no_punctuation = re.sub(token, ' ', text1)  # 去掉标点
    text_list = list(jieba.cut(str_no_punctuation))   # 分词列表
    text_list = [item for item in text_list if item != ' ' and item not in stopwords] # 去掉空格和停用词
    return ' '.join(text_list)


def load_raw_datasets():    
    labels = []
    texts = []
    base_dir = 'CN_Corpus/SogouC.reduced/Reduced'
    t1 = time.time()
    for cate_index, label in enumerate(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label)
        file_list = os.listdir(label_dir)
        labels_index[label] = cate_index # 记录分类标签的整数标号
        print("label: {}, len: {}".format(label, len(file_list)))

        for fname in file_list:
            f = open(os.path.join(label_dir, fname), encoding='gb2312', errors='ignore')
            texts.append(preprocess(f.read()))
            f.close()
            labels.append(labels_index[label])
            
    t2 = time.time()
    tm_cost = t2-t1
    print('\nDone. {} total categories, {} total docs. cost {} seconds.'.format(len(os.listdir(base_dir)), len(texts), tm_cost))
    return texts, labels

def load_pre_trained():
    # load pre-trained embedding model
    embeddings_index = {}
    with open('Embedding/sgns.sogou.word') as f:
        _, embedding_dim = f.readline().split()    
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors, dimension %s' % (len(embeddings_index), embedding_dim))
    return embeddings_index