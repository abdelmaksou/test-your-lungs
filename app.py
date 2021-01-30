#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect
import os

# Load various imports 
from datetime import datetime
from os import listdir
from os.path import isfile, join

import librosa
import librosa.display

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import keras
import tensorflow as tf


# Set flask app
app = Flask(__name__)

# Extract MFCCs
max_pad_len = 862 

def extract_features(file):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast', duration= 20) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        print("done")
    except Exception as e:
        print("Error encountered while parsing file: ", file.filename)
        return None 
     
    return mfccs

# Predict the disease
def predict(file):
    features = [] 
    data = extract_features(file)
    features.append(data)
    print('Finished feature extraction from ', len(features), ' files')

    features = np.array(features) # convert to numpy array

    # add channel dimension for CNN
    features = np.reshape(features, (*features.shape,1))

    # upload model
    model = tf.keras.models.load_model ('model/respiratory-sound-model.h5')

    c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']
    n_classes=6 # number of classes

    preds = model.predict(features) # label scores 

    # predict result
    pred_list = [float(np.array_str(preds).split()[0][2:])]
    for i in range(1,5):
        pred_list.append(float(np.array_str(preds).split()[i]))
    pred_list.append(float(np.array_str(preds).split()[5][:len(np.array_str(preds).split()[5])-2]))

    result = ""  
    i = pred_list.index(max(pred_list))
    result += c_names[i] + " : " + str(pred_list[i]/sum(pred_list)*100) + " %"
    return result

# main
@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        print('file uploaded successfully')

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            result = predict(file)
        return render_template('ui.html', request="POST", result=result)
    else:
        return render_template("ui.html")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)