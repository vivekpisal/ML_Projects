from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow_hub as hub
from flask import Flask,render_template,request
import pickle as pkl
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf 
from werkzeug.utils import secure_filename


app=Flask(__name__)

#with open('model.pkl','rb') as file:
#	model=pkl.load(file)


MODEL_PATH = 'model_resnet50.h5'

# Load your trained model
from tensorflow.keras.models import load_model
model = tf.keras.models.load_model('vgg16.h5')



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    
    
    x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.asscalar(np.argmax(preds, axis=1))
    '''if preds==0:
        preds="The Car IS Audi"
    elif preds==1:
        preds="The Car is Lamborghini"
    else:
        preds="The Car Is Mercedes"'''
    print(preds)
    return str(preds)



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None




@app.route("/base",methods=["POST","GET"])
def base():
	if request.method=="POST":
		year=int(request.form['Year'])	
		present_price=int(request.form['present_price'])
		km_driven=int(request.form['km_driven'])
		fuel_type=int(request.form['fuel_type'])
		seller_type=int(request.form['seller_type'])
		Transmission=int(request.form['Transmission'])
		owner=int(request.form['owner'])
		arr=[[year,present_price,km_driven,fuel_type,seller_type,Transmission,owner]]
		y_pred=model.predict(arr)
		y_pred=int(y_pred[0])
		return render_template("result.html",y_pred=y_pred)
	else:
		return render_template("form.html")




if __name__ == '__main__':
	app.run(debug=True)