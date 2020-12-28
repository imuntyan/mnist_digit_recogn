from flask import Flask, request, Response
import numpy as np
from urllib.request import urlopen
import cv2
import jsonpickle

app = Flask(__name__)


@app.route("/")
def index():
    return Response("It works!"), 200


@app.route("/abc")
def abc():
    return Response("abc"), 201


@app.route('/api/image', methods=['POST'])
def image():
    content = request.json
    data = content["data"]
    print(data)
    with urlopen(data) as response:
        dt = response.read()
        # convert string of image data to uint8
        nparr = np.fromstring(dt, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=True)