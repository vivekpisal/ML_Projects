from flask import Flask,request,render_template
import nltk
import pickle as pkl
from flask_mail import Mail,Message
from sklearn.feature_extraction.text import CountVectorizer

app=Flask(__name__)


app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = '465'
app.config['MAIL_USERNAME'] = 'vivekspisal235@gmail.com'
app.config['MAIL_PASSWORD'] = '9820987710'
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

with open('model2.pkl','rb') as f:
	model=pkl.load(f)

with open('cv.pkl','rb') as f:
	cv=pkl.load(f)


@app.route("/",methods=["GET","POST"])
def spam():
	if request.method=="GET":
		return render_template('form.html')
	else:
		msg=request.form['msg']
		value=[msg]
		cs=CountVectorizer(max_features=1000)
		X=cv.transform(value).toarray()
		y_pred=model.predict(X)
		return render_template('result.html',y_pred=y_pred)



@app.route("/tellreviews",methods=['GET','POST'])
def sendmail():
	if request.method=='POST':
		name=request.form['name']
		review=request.form['review']
		mail.send_message(name,
			sender ='vivekspisal235@gmail.com',
			recipients = ['codingthunder23@gmail.com'],
			body=review)
		return render_template('confirm_review.html')
	else:
		return render_template('reviews.html')
	    



if __name__=="__main__":
	app.run(debug=True) 
