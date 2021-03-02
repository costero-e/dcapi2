from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import glob
import os
import pathlib

from numpy import loadtxt
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import urllib.request
import sys, getopt
import urllib

def return_prediction(model, sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    batch_size = 32
    img_height = 180
    img_width = 180
    
    class_names = ['aca', 'n', 'scc']
       
    # pagineta_dir = tf.keras.utils.get_file('lung', origin=sample_json['pagineta'])
    # data_dir = pathlib.Path(pagineta_dir)
    
    # print(data_dir)
    
    unaimatge = sample_json['pagineta']
   
    data_dir = tf.keras.utils.get_file('fotolung', origin=unaimatge)
    
    img = keras.preprocessing.image.load_img(
    data_dir, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    message = "This image most likely belongs to {} with a {:.2f} percent confidence."
    print(message.format(class_names[np.argmax(score)], 100 * np.max(score)))
    
    return message.format(class_names[np.argmax(score)], 100 * np.max(score))

def loadImage(_URL):
    with urllib.request.urlopen(_URL) as url:
        # pujada = keras.preprocessing.image.load_img(BytesIO(url.read()), target_size=(180, 180))
        pujada = keras.preprocessing.image.load_img(url.read(), target_size=(180, 180))
    return keras.preprocessing.image.img_to_array(pujada)
    

app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'

model = load_model("lung_2.h5")

# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class LungForm(FlaskForm):
    pagineta = TextField('pagineta')

    submit = SubmitField('Analyze')
    
@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = LungForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['pagineta'] = form.pagineta.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['pagineta'] = str(session['pagineta'])

    results = return_prediction(model=model,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)
    
