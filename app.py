from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)


@app.route('/has-alzheimer')
def has_alzheimer():
    if request.data:
        json_string = request.get_json()
        random_forest_model = joblib.load('RandomForestClassifierModel.sav')
        prediction = random_forest_model.predict([[json_string['sex'], json_string['ses'], json_string['CDR']]])
        if prediction == 1:
            return jsonify('probably has alzheimer')
        else:
            return jsonify('probably has not alzheimer')


if __name__ == '__main__':
    app.run()
