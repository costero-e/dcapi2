from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

from flask import url_for, redirect, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from werkzeug.utils import secure_filename

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageOps
import sys
import os
import requests
from io import BytesIO
import os, glob

from keras.models import load_model
from keras.utils import normalize

from numpy import loadtxt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import urllib.request
import sys, getopt
import random

filename = "unknown"
folderPath = os.getcwd() + '/'
print("folderPath:", folderPath)



def image_to_array(image_path):
    im = Image.open(image_path)
    return np.array(im, dtype=float)

def label_decoder(key):
    label_mapping = {
        "1" : "aca",
        "2" : "n",
        "3" : "scc",
    }
    return label_mapping[key]

def run_model(data):

    loaded_model = load_model('gear_classifier_2.model')

    result_array = loaded_model.predict(data).tolist()[0]
    
    index = result_array.index(max(result_array)) + 1
    
    predicted_label = label_decoder(str(index))
    
    return predicted_label

   
def resize(image_name, dirPath):
    
    image_path = folderPath + image_name
    im = Image.open(image_path)
        
    desired_size = 128
    old_size = im.size 

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)
    
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(im, padding)

    filename, file_extension = os.path.splitext(image_name)
    new_filename = filename + "_resized.jpeg"
    new_im.save(folderPath + new_filename, "JPEG")
    
    return new_filename

def equalize_image(image_name, dirPath): 
    
    image_path = folderPath+ image_name
    im = Image.open(image_path)
    

    im_out = ImageOps.equalize(im)
    
    filename, file_extension = os.path.splitext(image_name) 
    new_filename = filename + "_equalized.jpeg"
    im_out.save(folderPath + new_filename, "JPEG")
    
    return new_filename


def return_prediction():
    
     
    list_of_files = glob.glob('static/*') 
    
    latest_file = max(list_of_files, key=os.path.getctime)
        
    resized_image_name = resize(latest_file, folderPath)
    equalized_image_name = equalize_image(resized_image_name, folderPath)
    
    path = os.path.abspath(equalized_image_name)
    
    numpy_data = image_to_array(path)
    numpy_data = numpy_data.reshape(1, 3, 128, 128)
    numpy_data = numpy_data.astype('float32')
    numpy_data = normalize(numpy_data)
    result_label = run_model(numpy_data)
    model = load_model('gear_classifier_2.model')
    predictions = model.predict(numpy_data)
    score = tf.nn.softmax(predictions)
    message = "Esta imagen pertenece a un tumor del tipo {} con una precisión de {:.2f} sobre 100."
    print(message.format(result_label, round(100 * np.max(score))))
    
    return message.format(result_label, round(100 * np.max(score)))
  
     

app = Flask(__name__)

app.config['SECRET_KEY'] = 'someRandomKey'

# http://wtforms.readthedocs.io/en/stable/fields.html
class UploadForm(FlaskForm):
    file = FileField()
    


@app.route('/', methods=['GET', 'POST'])
def index():
    global filename
    form = UploadForm()
    
    if form.validate_on_submit():
        files = glob.glob('static/*') 
        for f in files:
            os.remove(f)
        filename = secure_filename(form.file.data.filename)
        print("filename1:")
        print(filename)
        form.file.data.save('static/' + filename)
        return redirect(url_for('prediction'))

    return render_template('dcapi2.htm', form=form)


@app.route('/prediction')
def prediction():
    global filename
    print("filename2:")
    print(filename)

    results = return_prediction()

    return render_template('prediction.htm',results=results,imagen=filename)


if __name__ == '__main__':
    app.run(debug=True)
    
