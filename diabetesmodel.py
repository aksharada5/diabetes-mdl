import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pickle
from flask import Flask,request,jsonify,render_template

app=Flask(__name__)
filed='diabetes_model'
loadea_dmodel=pickle.load(open(filed,'rb'))
@app.route('/')
def home():
    return render_template('diab.html')
@app.route('/predict',methods=['POST'])
def predict():
    feat=[x for x in request.form.values()]
    fin_feat=[np.array(feat)]
    ot=loadea_dmodel.predict(fin_feat)
    pred=round(ot[0],2)
    if(pred==1):
        return render_template('diab.html',prediction="You have a risk of Diabetes")
    else:
         return render_template('diab.html',prediction="You do not have a risk of Diabetes")
app.run(debug=True)
