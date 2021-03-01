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
import sys, getopt

def return_prediction(model, sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    batch_size = 32
    img_height = 180
    img_width = 180
    
    class_names = ['aca', 'n', 'scc']
    
    img = keras.preprocessing.image.load_img(
    data_dir, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    message = print(
     "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    
    return message


    

app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'

model = load_model("lung_2.h5")

@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = LungForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['url'] = form.url.data

        return redirect(url_for("prediction"))


    return render_template('home.htm', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['url'] = tf.keras.utils.get_file('lung', origin=session['url'])

    results = return_prediction(model=model,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)
    
