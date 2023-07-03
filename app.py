from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_quality():
    pH = float(request.form.get('pH'))
    Temperature = int(request.form.get('Temperature'))
    Taste = int(request.form.get('Taste'))
    Odor = int(request.form.get('Odor'))
    Fat = int(request.form.get('Fat'))
    Turbidity = int(request.form.get('Turbidity'))
    Colour = int(request.form.get('Colour'))
    #profile_score = int(request.form.get('profile_score'))

    # prediction
    result = model.predict(np.array([pH,Temperature,Taste,Odor,Fat,Turbidity,Colour]).reshape(1,7))

    if result[0] == 0:
        result = 'High Quality'
    elif result[0] == 1:
        result = 'Low Quality'
    else:
        result = 'Medium Quality'

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)