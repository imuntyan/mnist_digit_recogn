from flask import Flask, request, Response
import numpy as np
from urllib.request import urlopen
import cv2
import jsonpickle
from .image_utils import imageprepare
from tensorflow import keras

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


        print(dt)
        # convert string of image data to uint8
        # nparr = np.fromstring(dt, np.uint8)

        digit = 0
        tva, new_image = imageprepare(dt)
        xnp = np.array(tva, dtype="float32").reshape((1,28, 28,1))
        model = keras.models.load_model('mnist_fitted_model')
        digit = model.predict(xnp)
        print(digit)

        nparr = np.fromstring(dt, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        response = {'message': 'image received. size={}x{}, digit={}'.format(img.shape[1], img.shape[0], digit)}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=True)