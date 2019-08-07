import flask
from flask import jsonify

from test_predict import predict
from test_predict_logistic import predictLR, predictWithSavedModel
from test_predict_tree import predictRF

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"


@app.route('/predict', methods=['GET'])
def predictRequest():
    predictPossibility = predictWithSavedModel()
    response = {
        "success": True,
        "predict": str(predictPossibility * 100) + '%'
    }
    return jsonify(response)

@app.route('/post', methods=['POST'])
def postRequest():
    return {'success': True}


app.run()
