# coding: utf-8

import os
import shutil
import re

import numpy as np 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix

from load_data import preprocess, load_datasets

X_train_data, y_train, X_test_data, y_test = load_datasets()

wordcloud = WordCloud(scale=4,
                      font_path='simhei.ttf',
                      background_color='white',
                      max_words = 100,
                      max_font_size = 60,
                      random_state=20).generate(X_train_data[1000])

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

stopwords = open('dict/stop_words.txt', encoding='utf-8').read().split()

# TF-IDF feature extraction
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_data)
words = tfidf_vectorizer.get_feature_names()

# naive bayes
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

news_lastest = ["360金融旗下产品有360借条、360小微贷、360分期。360借条是360金融的核心产品，是一款无抵押、纯线上消费信贷产品，为用户提供即时到账贷款服务（通俗可以理解为“现金贷”）用户借款主要用于消费支出。从收入构成来看，360金融主要有贷款便利服务费、贷后管理服务费、融资收入、其他服务收入等构成。财报披露，营收增长主要是由于贷款便利化服务费、贷款发放后服务费和其他与贷款发放量增加相关的服务费增加。",
                "检方并未起诉全部涉嫌贿赂的家长，但起诉名单已有超过50人，耶鲁大学、斯坦福大学等录取率极低的名校涉案也让该事件受到了几乎全球的关注，该案甚至被称作美国“史上最大招生舞弊案”。",
                "俄媒称，目前尚不清楚特朗普这一言论的指向性，因为近几日，伊朗官员们都在表达力图避免与美国发生军事冲突的意愿。5月19日早些时候，伊朗革命卫队司令侯赛因·萨拉米称，伊朗只想追求和平，但并不害怕与美国发生战争。萨拉米称，“我们（伊朗）和他们（美国）之间的区别在于，美国害怕发生战争，缺乏开战的意志。”"]
X_new_data = [preprocess(doc) for doc in news_lastest]
X_new_tfidf = tfidf_vectorizer.transform(X_new_data)


predicted = classifier.predict(X_new_tfidf)

# Pipeline
text_clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(X_train_data, y_train)
# text_clf.predict(X_new_data)

predicted = text_clf.predict(X_test_data)
print(classification_report(predicted, y_test))
# confusion_matrix(predicted, y_test)

# Logistic Regression
text_clf_lr = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', LogisticRegression()),
])
text_clf_lr.fit(X_train_data, y_train)
# text_clf_lr.predict(X_new_data)
predicted_lr = text_clf_lr.predict(X_test_data)
print(classification_report(predicted_lr, y_test))
# confusion_matrix(predicted_lr, y_test)

# SVM
text_clf_svm = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2')),
])
text_clf_svm.fit(X_train_data, y_train)
# text_clf_svm.predict(X_new_data)
predicted_svm = text_clf_svm.predict(X_test_data)
print(classification_report(predicted_svm, y_test))
# confusion_matrix(predicted_svm, y_test)