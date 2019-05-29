# coding: utf-8

import os
import re
import jieba
import time
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Embedding, Input
from keras.layers import Conv1D, MaxPooling1D, Flatten

from load_data import labels_index, load_raw_datasets, load_pre_trained
from utils import plot_history

texts, labels = load_raw_datasets()
embeddings_index = load_pre_trained()

MAX_SEQUENCE_LEN = 1000  # sequence length
MAX_WORDS_NUM = 20000  # max words
VAL_SPLIT_RATIO = 0.2 # ratio for validation
EMBEDDING_DIM = 300 # embedding dimension

# process datasets by keras API

tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print(len(word_index)) # all token found

dict_swaped = lambda _dict: {val:key for (key, val) in _dict.items()}
word_dict = dict_swaped(word_index) # swap key-value
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN)

labels_categorical = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels_categorical.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels_categorical = labels_categorical[indices]

# split data by ratio
val_samples_num = int(VAL_SPLIT_RATIO * data.shape[0])
x_train = data[:-val_samples_num]
y_train = labels_categorical[:-val_samples_num]
x_val = data[-val_samples_num:]
y_val = labels_categorical[-val_samples_num:]

# generate embedding matrix
embedding_matrix = np.zeros((MAX_WORDS_NUM+1, EMBEDDING_DIM)) # row 0 for 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < MAX_WORDS_NUM:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

# build models

# model 1 without pre-trained embedding

input_dim = x_train.shape[1]
model1 = Sequential()
model1.add(Embedding(input_dim=MAX_WORDS_NUM+1, 
                    output_dim=EMBEDDING_DIM, 
                    input_length=MAX_SEQUENCE_LEN))
model1.add(Flatten())
model1.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(len(labels_index), activation='softmax'))
model1.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history1 = model1.fit(x_train, 
                    y_train,
                    epochs=30,
                    batch_size=128,
                    validation_data=(x_val, y_val))

plot_history(history1)

# model 2 with pre-trained embedding

model2 = Sequential()
model2.add(Embedding(input_dim=MAX_WORDS_NUM+1, 
                    output_dim=EMBEDDING_DIM, 
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LEN,
                    trainable=False))
model2.add(Flatten())
model2.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(len(labels_index), activation='softmax'))

model2.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history2 = model2.fit(x_train, 
                    y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(x_val, y_val))

plot_history(history2)


# model 3 with CNN

embedding_layer = Embedding(input_dim=MAX_WORDS_NUM+1,
                            output_dim=EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LEN,
                            trainable=False)


sequence_input = Input(shape=(MAX_SEQUENCE_LEN,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model3 = Model(sequence_input, preds)
model3.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

history3 = model3.fit(x_train, 
                    y_train,
                    epochs=6,
                    batch_size=128,
                    validation_data=(x_val, y_val))

plot_history(history3)



