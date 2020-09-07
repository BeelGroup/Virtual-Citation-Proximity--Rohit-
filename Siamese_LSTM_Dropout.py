# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 12:21:42 2020

@author: Rohit
"""

import matplotlib.pyplot as plt
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Embedding, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pd.set_option('display.max_columns',20)
#preparing Dataset using title-paragraph.csv and citation data
title_paragraph = pd.read_csv('title_paragraph.csv')
col_names = ['hash', 'title_a', 'title_b', 'dist', 'count', 'page_idA', 'page_idB', 'cpi']
citation_pairs = pd.read_csv('citationdata5000000.csv', sep = '|',nrows = 1000000, header = None, names = col_names)
citation_pairs = pd.read_csv('processeddata.csv')
citation_pairs = citation_pairs[citation_pairs['dist'] <= 1000]
citation_pairs = citation_pairs.dropna(axis = 0)
text_a = pd.DataFrame(citation_pairs['title_a'])
text_a.rename(columns={'title_a':'title'}, inplace = True)
text_a_data = pd.merge(text_a, title_paragraph, how = 'left', on = 'title')
citation_pairs['text_a'] = text_a_data['text']
text_b = pd.DataFrame(citation_pairs['title_b'])
text_b.rename(columns={'title_b':'title'}, inplace = True)
text_b_data = pd.merge(text_b, title_paragraph, how = 'left', on = 'title')
citation_pairs['text_b'] = text_b_data['text']
citation_pairs = citation_pairs.dropna(axis = 0)
#Cleaning Data
def rephrase(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
def stripunct(data): 
    return re.sub('[^A-Za-z]+', ' ', str(data), flags=re.MULTILINE|re.DOTALL)
stop_words = set(stopwords.words('english'))
stemm = WordNetLemmatizer()
def compute(sent): 
    
    sent = rephrase(sent) 
    sent = stripunct(sent) 
    
    words=word_tokenize(str(sent.lower())) 
    
    #Removing all single letter and and stopwords from question 
    sent1=' '.join(str(stemm.lemmatize(j)) for j in words if j not in stop_words and (len(j)!=1)) 
    return sent1
clean_stemmed_text1 = []
clean_stemmed_text2 = []
combined_stemmed_text = []
for _, row in tqdm(citation_pairs.iterrows()):
    csq1= compute(row['text_a'])
    csq2= compute(row['text_b'])
    clean_stemmed_text1.append(csq1)
    clean_stemmed_text2.append(csq2)
    combined_stemmed_text.append(csq1+" "+csq2)
citation_pairs['clean_stemmed_text1'] = clean_stemmed_text1
citation_pairs['clean_stemmed_text2'] = clean_stemmed_text2
citation_pairs['combined_stemmed_text'] = combined_stemmed_text
#Tokenization of text
token = Tokenizer()
token.fit_on_texts(citation_pairs['combined_stemmed_text'].values)
X_train, X_test, y_train, y_test = train_test_split(citation_pairs[['title_a','title_b','clean_stemmed_text1', 'clean_stemmed_text2']].tail(100000), citation_pairs['dist'].tail(100000), test_size=0.2, random_state=100)
X_train, X_val, y_train, y_val = train_test_split(X_train[['clean_stemmed_text1', 'clean_stemmed_text2']], y_train, test_size=0.2, random_state=100)
train_text1_seq = token.texts_to_sequences(X_train['clean_stemmed_text1'].values)
train_text2_seq = token.texts_to_sequences(X_train['clean_stemmed_text2'].values)
val_text1_seq = token.texts_to_sequences(X_val['clean_stemmed_text1'].values)
val_text2_seq = token.texts_to_sequences(X_val['clean_stemmed_text2'].values)
test_text1_seq = token.texts_to_sequences(X_test['clean_stemmed_text1'].values)
test_text2_seq = token.texts_to_sequences(X_test['clean_stemmed_text2'].values)
max_len = 50
train_text1_seq = pad_sequences(train_text1_seq, maxlen=max_len, padding='post')
train_text2_seq = pad_sequences(train_text2_seq, maxlen=max_len, padding='post')
val_text1_seq = pad_sequences(val_text1_seq, maxlen=max_len, padding='post')
val_text2_seq = pad_sequences(val_text2_seq, maxlen=max_len, padding='post')
test_text1_seq = pad_sequences(test_text1_seq, maxlen=max_len, padding='post')
test_text2_seq = pad_sequences(test_text2_seq, maxlen=max_len, padding='post')
#Creating Embedding Matrix
embeddings_index = {}
with open('glove.840B.300d.txt', encoding = 'UTF-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0] # The word
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients
not_present_list = []
vocab_size = len(token.word_index) + 1
print('Loaded %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((vocab_size, len(embeddings_index['no'])))
for word, i in token.word_index.items():
    if word in embeddings_index.keys():
        embedding_vector = embeddings_index.get(word)
    else:
        not_present_list.append(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.zeros((vocab_size, len(embeddings_index['no'])))
np.save('embedding_matrix.npy', embedding_matrix)
embedding_matrix = np.load('embedding_matrix.npy')
SC = StandardScaler()
y_train = SC.fit_transform(y_train.values.reshape(-1,1))
y_val = SC.fit_transform(y_val.values.reshape(-1,1))
y_test = SC.fit_transform(y_test.values.reshape(-1,1))
input_1 = Input(shape=(train_text1_seq.shape[1],))
input_2 = Input(shape=(train_text2_seq.shape[1],))
embed = Embedding(input_dim = vocab_size, 
                  output_dim=300,weights=[embedding_matrix], 
                  input_length=train_text1_seq.shape[1],trainable=False) 
lstm_1 = embed(input_1)
lstm_2 = embed(input_2)
lstm = LSTM(50, return_sequences = True, activation = 'relu', dropout = 0.2)
vector_1 = lstm(lstm_1)
vector_2 = lstm(lstm_2)
vector_1 = Flatten()(vector_1)
vector_2 = Flatten()(vector_2)
conc = concatenate([vector_1,vector_2])
out = Dense(1)(conc)
model = Model([input_1, input_2], out)
callback = [EarlyStopping(patience = 8)]
model.compile(optimizer=Adam(0.00001), loss='mse', metrics=['mae'])
model.load_weights('vcp8.h5')
history = model.fit([train_text1_seq,train_text2_seq],y_train.values.reshape(-1,1), epochs = 100,
              batch_size=32,validation_data=([val_text1_seq, val_text2_seq],y_val.values.reshape(-1,1)),
              callbacks = callback)
model.save_weights('vcp9.h5')
result = model.predict([test_text1_seq,test_text2_seq])
result = pd.DataFrame(result)
ytest = pd.DataFrame(y_test)
loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
mae_history_val = history.history['val_loss']
mae_history = history.history['loss']
plt.plot(epochs, mae_history,label='Training loss')
plt.plot(epochs, mae_history_val,label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

mae_history_val = history.history['val_mae']
mae_history = history.history['mae']
plt.plot(epochs, mae_history,label='Training mae')
plt.plot(epochs, mae_history_val,label='Validation mae')
plt.axhline(np.mean(mae_history), color = 'black', label = 'Mean MAE')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()