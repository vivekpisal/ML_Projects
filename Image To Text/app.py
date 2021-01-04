from flask import Flask,request,render_template
import os
from model import ImageToText 

# Flask config
app = Flask(__name__)
UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/',methods=['GET','POST'])
def home():
	if request.method=='POST':
		file=request.files['myfile']
		file.save(f'static/{file.filename}')
		op=ImageToText(f'static/{file.filename}')
		os.remove(f'static/{file.filename}')
		return render_template('prediction.html',op=op)
	else:
		return render_template('home.html')


if __name__=='__main__':
	app.run(debug=True)