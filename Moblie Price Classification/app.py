from flask import Flask,request,render_template
import pickle


app=Flask(__name__)


@app.route("/",methods=["POST","GET"])
def home():
	if request.method=='POST':
		with open('model1.pkl','rb') as f:
			model=pickle.load(f)
		ram=int(request.form['ram'])
		ram=ram*1000
		px_height=int(request.form['px_height'])
		px_width=int(request.form['px_width'])
		battery_power=int(request.form['battery_power'])
		mobile_wt=int(request.form['mobile_wt'])
		int_memory=int(request.form['int_memory'])
		y_pred=model.predict([[ram,px_height,px_width,battery_power,mobile_wt,int_memory]])
		return render_template('result.html',y_pred=y_pred)
	return render_template('index.html')


if __name__=="__main__":
	app.run(debug=True)