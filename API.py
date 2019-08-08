import flask
from flask import jsonify, request

from test_predict import predict
from predict_binary_logistic import predictLR, predictWithSavedModel
from predict_nonbinary_logistic import predictNonBinaryLR
from test_predict_tree import predictRF

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"


@app.route('/predict', methods=['GET'])
def predictRequest():
    #predictPossibility = predictWithSavedModel()
    predictPossibility = predictLR()
    response = {
        "success": True,
        "predict": str(predictPossibility * 100) + '%'
    }
    return jsonify(response)

@app.route('/predict-2', methods=['GET'])
def predictNonBinaryRequest():
    predictPossibility = predictNonBinaryLR()
    response = {
        "success": True,
        "predict": str(predictPossibility * 100) + '%'
    }
    return jsonify(response)

@app.route('/post', methods=['POST'])
def postRequest():
    return {'success': True}


app.run()
