#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 22:30:07 2022

@author: mdsanowarhossain
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle



app=Flask(__name__)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)


@app.route('/')
def welcome():
    return 'welcome All'

@app.route('/predict')
def predict_note_authentication():
    
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return 'The predicted value is'+str(prediction)

"""
http://127.0.0.1:5000/predict?variance=2&skewness=3&curtosis=2&entropy=1
"""


@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    
    df_test=pd.read_csv(request.files.get('file'))
    prediction=classifier.predict(df_test)
    return 'The prediction value for the csv is'+str(list(prediction))
    



if __name__=='__main__':
    app.run()
    
    