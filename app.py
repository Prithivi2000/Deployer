from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index - copy.html')


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if output > str(0.5):
        return render_template('index - copy.html', pred='Customer will be defaulty in next month.')
    else:
        return render_template('index - copy.html', pred='Customer will not be defaulty in next month.')


app.run()