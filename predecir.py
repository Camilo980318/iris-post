#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import numpy as np
from sklearn.externals import joblib
import sklearn

app = Flask(__name__)

@app.route('/', methods=['POST'])
def clasificar():
    #cargar el modelo de clasificacion 
    knn = joblib.load("modelo_knn.mod")
     
    json_data = request.json
    print(type(json_data))
    
    sl = json_data['sl']
    sw = json_data['sw']
    pl = json_data['pl']
    pw = json_data['pw']
    
    datos = np.array([sl,sw,pl,pw], ndmin = 2)
    
    prediction = knn.predict(datos)
    
    print(prediction)
    
    respuesta =  {"sl":sl,"sw":sw,"pl":pl,"pw":pw,"Clase":prediction[0]} 
    
    respuesta2 = str(respuesta)
    
    return respuesta2

if __name__ == '__main__':
    app.run()
