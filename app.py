

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#entrainement du modèle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#base d'apprentissage et base de test
from sklearn.model_selection import train_test_split
 
import pickle

from flask import Flask, jsonify, request, render_template
import os

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

assurance = pd.read_csv('assurance.csv')

assurance.loc[(assurance.sex == "female"), "sex"] = 0
assurance.loc[(assurance.sex == "male"), "sex"] = 1

assurance.loc[(assurance.smoker == "yes"), "smoker"] = 1
assurance.loc[(assurance.smoker == "no"), "smoker"] = 0

assurance.loc[(assurance.region == "northeast"), "region"] = 1
assurance.loc[(assurance.region == "southeast"), "region"] = 2
assurance.loc[(assurance.region == "southwest"), "region"] = 3
assurance.loc[(assurance.region == "northwest"), "region"] = 4

assurance[["sex", "smoker", "region"]] = assurance[["sex", "smoker", "region"]].apply(pd.to_numeric)

@app.route('/learning')
def learning():
	X = pd.DataFrame(np.c_[assurance['smoker'],assurance['age'],assurance['bmi']], columns = ['smoker','age','bmi'])
	Y = assurance['charges']

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
	reg_assurance = LinearRegression()
	reg_assurance.fit(X_train, Y_train)

	pickle.dump(reg_assurance, open('model.pkl','wb'))

	return 'Learning is done'


@app.route('/predict', methods=['POST'])
def predict():
	#get param
	smoker = int(request.form["smoker"])
	age = int(request.form["age"])
	size = (int(request.form["size"])/100)
	weight = int(request.form["weight"])
	bmi = round(weight/(size * size),3) # arrondir au centiéme
	print(bmi)
	model = pickle.load(open('model.pkl','rb'))
	result = model.predict([[smoker,age,bmi]])
	return render_template('index.html', prediction_text='Le coût de l\'assurance sera de  {} $'.format(round(result[0],2)))

@app.route('/')
def index():
	return render_template('index.html', prediction_text='Compléter le formulaire pour avoir une prédiction')


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)

