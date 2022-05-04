from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# load the model from disk
filename = 'nlp_model.pkl'
model = pickle.load(open(filename, 'rb'))

#filename = 'my_model.pkl'
#lstm_model = pickle.load(open(filename, 'rb'))

cv = pickle.load(open('tranform.pkl', 'rb'))
#load datasets
x_train = pickle.load(open('X_train.pkl', 'rb'))
y_train = pickle.load(open('y_train.pkl', 'rb'))

x_test = pickle.load(open('X_test.pkl', 'rb'))
y_test = pickle.load(open('y_test.pkl', 'rb'))

max_vocab_size = 1000
tokenizer = Tokenizer(num_words=max_vocab_size)

#start flask app
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]

		#vect = tokenizer.texts_to_sequences([data])
		#vect_padded = tf.keras.preprocessing.sequence.pad_sequences(vect, maxlen=400)
		#my_prediction = lstm_model.predict(vect_padded)
		#my_prediction = my_prediction.reshape(my_prediction.shape[0])
		#if my_prediction >= 0.5:
		#	my_prediction = 1
		#if my_prediction < 0.5:
		#	my_prediction = 0

		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
		test_score = model.score(x_test, y_test)
		#train_score = model.score(x_train, y_train)


	return render_template('result.html',prediction = my_prediction, test_accuracy= test_score )



if __name__ == '__main__':
	app.run(debug=True)