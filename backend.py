# reference: https://zhuanlan.zhihu.com/p/100752901

from flask import Flask
from flask import request
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import json
import numpy as np

from trading_bot.agent import Agent

app = Flask(__name__)

#load model
model_name = "models/doubledqn_goog.h5"
agent = load_model(model_name)

window_size = 10
strategy = 'dqn'
pretrained = True

def get_stock_data(data):
    if type(data) is np.ndarray:
        raise Exception("Numpy input is required")
    try:
        return data[:window_size]
    except:
        print("input sequence requires minimal length of {}, while current shape is {}".format(window_size, data.size()))

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route("/predict", methods=["GET","POST"])
def predict():
    data = request.get_json()
    if 'bid_price' not in data:
        return {'result': 'no price input'}
    state = data['bid_price']
    
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    action = agent.act(state, is_eval=True)
    
    # if action == 1:
    #         agent.inventory.append(data[t])
        
    # # SELL
    # elif action == 2 and len(agent.inventory) > 0:
    #     bought_price = agent.inventory.pop(0)
    #     delta = data[t] - bought_price
    #     reward = delta #max(delta, 0)
    #     total_profit += delta
        
    # # HOLD
    # else:
    #     pass
    
    return {'result':action.tolist()}

if __name__ == '__main__':
    app.run()