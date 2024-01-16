from flask import Flask, request, escape, jsonify, url_for,Blueprint,render_template
from flask_cors import CORS
import json


app = Flask(__name__)
CORS(app)

from videoToText import video
app.register_blueprint(video)

@app.route("/")
def hello():
    return jsonify({"message":"hello to our app"})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')