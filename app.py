
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow
# import cv2

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
# from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
# run_with_ngrok(app)

# Model saved with Keras model.save()
MODEL_PATH ='steel_defect.h5'

# Load your trained model
model = load_model(MODEL_PATH)





def model_predict(file, model):
     img = image.load_img(file, grayscale = True ,target_size=(200, 200))
     x = image.img_to_array(img)
     x = np.array(x)
     x = np.expand_dims(x, axis=0)
     x = x.astype('float32')/255
     preds = model.predict(x)
     preds = np.argmax(preds)
     labels = ['Crazing','Inclusion','Patches','Pitted','Rolled','Scratches']
     preds = labels[preds]
     preds =print("The Metal has {} type of Surface Defect".format(preds))
     return preds
   
   

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
     if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']

        # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)
        # Make prediction
        preds = model_predict(file, model)
        result=preds
        return result
     return None


if __name__ == '__main__':
    app.run()
