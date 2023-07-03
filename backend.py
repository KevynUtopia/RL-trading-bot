# reference: https://zhuanlan.zhihu.com/p/100752901

from flask import Flask
from flask import request
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import json

app = Flask(__name__)

#load model
path = "models/doubledqn_goog.h5"
model = load_model(path)

#处理成相同序列
def get_sequence(text):
    x_token = tokenizer.texts_to_sequences(text)
    x_processed = sequence.pad_sequences(x_token, maxlen=100, value=0)
    return x_processed

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route("/predict", methods=["GET","POST"])
def predict():
    data = request.get_json()
    if 'values' not in data:
        return {'result': 'no input'}
    arr = data['values']
    res = model.predict(get_sequence(arr))
    return {'result':res.tolist()}

if __name__ == '__main__':
    app.run()