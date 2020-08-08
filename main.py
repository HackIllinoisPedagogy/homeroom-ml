from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from tutor import Tutor


app = Flask(__name__)
CORS(app)

soln = open("solution.txt").read()
tutor = Tutor(3, soln=soln)


@app.route("/get_hint")
@cross_origin()
def get_hint():
    curr_state = request.args.get("state")
    hint = tutor.get_hint(curr_state)
    return jsonify({"hint": hint})


@app.route("/reset_state")
@cross_origin()
def reset_state():
    tutor.reset_user_state()
    return jsonify({"worked": True})


@app.route("/load_soln")
@cross_origin()
def load_soln():
    soln = request.args.get("soln")
    tutor.load_soln(soln)
    return jsonify({"worked": True})


if __name__ == '__main__':
    app.run(debug=True)
