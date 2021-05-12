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
# from fileinput import filename
# import urllib

filename = "unknown"
# folderPath = '/Users/barnatasa/Desktop/Nou3/'
folderPath = os.getcwd() + '/'
print("folderPath:", folderPath)



def image_to_array(image_path):
    """
    Input: Image path
    Output: Flatten array of 3x128x128 image pixels in range[0,255]
    """
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
    """
    Load Keras model (.yaml) and weights (.h5), and predict image label
    Input: Image pixels array
    Output: Predicted label
    """
    
    # Load Keras Model
    loaded_model = load_model('gear_classifier_2.model')

    # Predict results array (array of 12 elements, one 1.0 and 11 are 0.0's)
    result_array = loaded_model.predict(data).tolist()[0]
    
    # Get the position of the element 1.0 within the array
    index = result_array.index(max(result_array)) + 1
    
    # Decode results
    predicted_label = label_decoder(str(index))
    
    return predicted_label

   
def resize(image_name, dirPath):
    """
    Pick a basic color (Black) and pad the images that do not have a 1:1 aspect ratio.
    Reshape without stretching to a 128x128 pixel array shape
    """
    
    image_path = folderPath + image_name
    im = Image.open(image_path)
        
    desired_size = 128
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)
    
    # create a new image and paste the resized on it
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

def equalize_image(image_name, dirPath): #"imagename_resized.JPEG"
    """
    Ensure for each image that the pixel range is [0,255] (constrast stretching)
    by applying the equalise method (normalise works also)
    """
    
    image_path = folderPath+ image_name
    im = Image.open(image_path)
    
    # Plotting histogram for resized image
    #im_array = np.array(im)
    #plt.hist(im_array.flatten(), bins=50, range=(0.0, 300))
    
    # Equalize image
    im_out = ImageOps.equalize(im)
    
    # Save equalized image
    filename, file_extension = os.path.splitext(image_name) 
    new_filename = filename + "_equalized.jpeg"
    im_out.save(folderPath + new_filename, "JPEG")
    
    return new_filename


def return_prediction():
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
       
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
    print('Predicted : ' + result_label + '\n')
    
    return 'Predicted : ' + result_label + '\n'
  
     

app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'

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

    results = return_prediction()

    return render_template('prediction.htm',results=results,imagen=filename)


if __name__ == '__main__':
    app.run(debug=True)
    
