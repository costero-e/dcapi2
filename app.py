from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

from flask import url_for, redirect, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from werkzeug.utils import secure_filename

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
# from fileinput import filename
# import urllib

filename = "unknown"

def return_prediction(model):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    batch_size = 32
    img_height = 180
    img_width = 180
    
    class_names = ['aca', 'n', 'scc']
    
    list_of_files = glob.glob('static/*') 
    
    latest_file = max(list_of_files, key=os.path.getctime)
    path = os.path.abspath(latest_file)
         
    img = keras.preprocessing.image.load_img(
    path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    message = "Esta imagen pertenece a un tumor del tipo {} con un {:.2f} por ciento de fiabilidad."
    print(message.format(class_names[np.argmax(score)], 100 * np.max(score)))
    
    return message.format(class_names[np.argmax(score)], 100 * np.max(score))

app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'

model = load_model("lung_2.h5")

# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class UploadForm(FlaskForm):
    file = FileField()
    


@app.route('/', methods=['GET', 'POST'])
def index():
    global filename
    form = UploadForm()
    
    if form.validate_on_submit():
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

    results = return_prediction(model=model)

    return render_template('prediction.htm',results=results,imagen=filename)


if __name__ == '__main__':
    app.run(debug=True)
    
