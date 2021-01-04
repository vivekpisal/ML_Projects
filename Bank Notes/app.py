from flask import Flask,render_template,request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

app=Flask(__name__)


@app.route('/',methods=['POST','GET'])
def note():
	if request.method=='GET':
		return render_template('form.html')
	else:
		df=pd.read_csv('BankNote_Authentication.csv')
		X=df.drop('class',axis=1)
		y=df['class']
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)
		model=LogisticRegression()
		model.fit(X_train,y_train)
		variance=float(request.form['variance'])
		skewness=float(request.form['skewness'])
		curtosis=float(request.form['curtosis'])
		entropy=float(request.form['entropy'])
		new=np.array([[variance,skewness,curtosis,entropy]])
		y_pred=model.predict(new)
		return render_template('form.html',y_pred=y_pred)



if __name__ == '__main__':
	app.run(debug=True)