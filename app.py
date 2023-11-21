import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np 
import pandas as pd 

# create flask app 
app = Flask(__name__)
# load the model 
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))
# Create API
@app.route('/')
def home():
  return render_template('home.html')
# create predict API
@app.route('/predict_api', methods= ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    print(np.array(list(data.value())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0]) 
if __name__ == "__main__":
  app.run(debug = True) 





























