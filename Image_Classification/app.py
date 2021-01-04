import numpy as np
import os
from flask import Flask,render_template,request
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename


app=Flask(__name__)
model=load_model("vgg19.h5")


def model_predict(img_path,model):
	img=image.load_img(img_path,target_size=(224,224))
	x=image.img_to_array(img)
	x=np.expand_dims(x,axis=0)
	x=preprocess_input(x)
	preds=model.predict(x)
	return preds


@app.route('/',methods=['GET','POST'])
def upload():
	if request.method=='POST':
		f=request.files['file']
		basepath=os.path.dirname(__file__)
		file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
		f.save(file_path)
		pred=model_predict(file_path,model)
		pred_class=decode_predictions(pred,top=1)
		result=str(pred_class[0][0][1])
		return result
	else:
		return render_template('index.html')





if __name__=="__main__":
	app.run(debug=True)