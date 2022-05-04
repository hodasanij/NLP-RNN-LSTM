from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import pickle

# load the model from disk
filename = 'nlp_model.pkl'
model = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

#load datasets
x_train = pickle.load(open('X_train.pkl', 'rb'))
y_train = pickle.load(open('y_train.pkl', 'rb'))

x_test = pickle.load(open('X_test.pkl', 'rb'))
y_test = pickle.load(open('y_test.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
		test_score = model.score(x_test, y_test)
	return render_template('result.html',prediction = my_prediction, test_accuracy= test_score)



if __name__ == '__main__':
	app.run(debug=True)