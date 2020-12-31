from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
import numpy as np
from urllib.request import urlopen
# import cv2
import jsonpickle
from .image_utils import imageprepare, ascii_rep
from tensorflow import keras

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def index():
    return Response("It works!"), 200


@app.route("/abc")
def abc():
    return Response("abc"), 201


@app.route('/api/image', methods=['POST'])
@cross_origin()
def image():
    content = request.json
    data = content["data"]
    with urlopen(data) as response:
        dt = response.read()

        tva, new_image = imageprepare(dt)

        print(ascii_rep(tva))

        xnp = np.array(tva, dtype="float32").reshape((1,28, 28,1))
        model = keras.models.load_model('fit/mnist_fitted_model')
        digits = model.predict(xnp)

        # nparr = np.frombuffer(dt, np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        response = {'message': digits.ravel().tolist()}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=True)