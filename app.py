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
# import pathlib

from numpy import loadtxt
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import urllib.request
import sys, getopt
import random
# import urllib

def return_prediction(model, sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    batch_size = 32
    img_height = 180
    img_width = 180
    
    class_names = ['aca', 'n', 'scc']
    
    pagineta_dir = tf.keras.utils.get_file('{:03}'.format(random.randrange(1, 10**10)),origin=sample_json['Enlace'])
      
    img = keras.preprocessing.image.load_img(
    pagineta_dir, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    message = "Esta imagen pertenece a un tumor del tipo {} con un {:.2f} por ciento de fiabilidad."
    print(message.format(class_names[np.argmax(score)], 100 * np.max(score)))
    
    return message.format(class_names[np.argmax(score)], 100 * np.max(score))

def return_image(sample_json):
    
    enlacen_imagen = sample_json['Enlace']
    
    return enlacen_imagen

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
    Enlace = TextField('Enlace')

    submit = SubmitField('Clasifica')
    
@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = LungForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['Enlace'] = form.Enlace.data

        return redirect(url_for("prediction"))


    return render_template('dcapi2.htm', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['Enlace'] = str(session['Enlace'])

    results = return_prediction(model=model,sample_json=content)
    
    imagenes = return_image(sample_json=content)

    return render_template('prediction.htm',results=results,imagenes=imagenes)


if __name__ == '__main__':
    app.run(debug=True)
    
