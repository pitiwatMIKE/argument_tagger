#falsk
from flask import Flask, render_template, jsonify, redirect, url_for
from flask.globals import request

#utils
import random
from INIT_TEXT import init_text

#call model
from use_model.model_CRF import call_model_CRF
from use_model.model_LSTM import call_model_LSTM
from use_model.model_BiLSTM import call_model_BiLSTM
from use_model.model_BiLSTM_CRF import call_model_BiLSTM_CRF

app = Flask(__name__, static_url_path="/static", static_folder='./static')

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/initinput', methods=['GET'])
def init_input():
    list_input = init_text()
    input_random = random.choice(list_input)

    result = {'init_input': input_random}
    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json["input_text"]
    model_name = request.json["model"]
    predict = ""

    #use model trained
    if model_name == "CRF":
        predict = call_model_CRF(input_text)
    elif model_name == "LSTM":
        predict = call_model_LSTM(input_text)
    elif model_name == "BiLSTM":
        predict = call_model_BiLSTM(input_text)
    elif model_name == "BiLSTM-CRF":
        predict = call_model_BiLSTM_CRF(input_text)
        
    result = {'predict': predict}
    return jsonify(result)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename))


if __name__ == "__main__":
    app.run(debug=False)