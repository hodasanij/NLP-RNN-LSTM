#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, Input
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np

from string import punctuation
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

df = pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
#print(tf.__version__)

# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Extract Feature With CountVectorizer
cv = CountVectorizer()

# Fit the Data
X = cv.fit_transform(X)
pickle.dump(cv, open('tranform.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
r = model.fit(X_train, y_train)
model.score(X_test, y_test)

#saving X_train and y_train into .pkl
pickle.dump(X_train, open('X_train.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))

#saving X_test and y_test into .pkl
pickle.dump(X_test, open('X_test.pkl', 'wb'))
pickle.dump(y_test, open('y_test.pkl', 'wb'))

print('Accuracy on training set: {}'.format(model.score(X_train, y_train)))
print('Accuracy on testing set: {}'.format(model.score(X_test, y_test)))

#filename = 'nlp_model.pkl'
#pickle.dump(model, open(filename, 'wb'))


##################################################################################
##################################################################################

#load data
data = pd.read_csv('comspam.csv')

df = data[['Body','Label']]


def text_cleaning(text):
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", str(text))
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r'http\S+', ' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # remove numbers

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Return a list of words
    return (text)

#clean the review
df['cleaned_review'] = df['Body'].apply(text_cleaning)

print(df['cleaned_review'][6043])
print(df['Label'][6043])
print(df.shape)

# to split dataset into test and training dataset
x_train, x_test, y_train, y_test = train_test_split(df['cleaned_review'].values, df['Label'].values, test_size=0.25)


# set the maximum number of words that can be use
max_vocab_size = 1000

# create tokenizer
tokenizer = Tokenizer(num_words = max_vocab_size)
tokenizer.fit_on_texts(x_train)

# convert text into sequence of integers
x_train_s = tokenizer.texts_to_sequences(x_train)
x_test_s = tokenizer.texts_to_sequences(x_test)

x_train_s_padded = tf.keras.preprocessing.sequence.pad_sequences(x_train_s, maxlen= 400)
x_test_s_padded = tf.keras.preprocessing.sequence.pad_sequences(x_test_s, maxlen= 400)

print(x_train[2])
print(x_train_s[2])

model = Sequential()
model.add(Input(shape=(400,)) )
model.add(Embedding(max_vocab_size, 300))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(100, activation="relu"))
model.add(Dense(1, activation='sigmoid'))

#define the optimizer, again just for testing
#optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

#early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 5)

#model.compile(optimizer = optimizer,
 #             loss = 'binary_crossentropy',
  #            metrics = ['accuracy'])

# using model.fit
#r = model.fit(x_train_s_padded, y_train,
#              batch_size = 60,
#             epochs = 20,
 #             validation_data = (x_test_s_padded, y_test),
  #            callbacks=[early_stopping])


#filename = 'my_model11.pkl'
#pickle.dump(model, open(filename, 'wb'))

filename = 'my_model.pkl'
lstm_model = pickle.load(open(filename, 'rb'))

tw = "you"
vect = tokenizer.texts_to_sequences([tw])
vect_padded = tf.keras.preprocessing.sequence.pad_sequences(vect, maxlen= 400)
my_prediction = lstm_model.predict(vect_padded )
my_prediction = my_prediction.reshape(my_prediction.shape[0])
print(my_prediction)
if my_prediction >= 0.5:
  my_prediction = 1
else:
  my_prediction = 0
print(my_prediction)
