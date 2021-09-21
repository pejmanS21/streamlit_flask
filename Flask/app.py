import matplotlib.pyplot as plt
from flask import Flask, request, Response, send_file
import numpy as np
import cv2
import jsonpickle
from unet import *

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, world'

@app.route('/<name>', methods=['GET', 'POST'])
def name(name):
    return f"you're using {name}!"

@app.route('/process', methods=['POST'])
def get_segmented():

    image = request_handler(request)
    model = unet(pretrained_weights='weigths/cxr_seg_unet.hdf5')
    mask = service(model, image)

    response = {'message': 'image received. size={} and {}, {}'.format(mask.shape, mask.min(), mask.max())}

    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


def request_handler(request):
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = img.reshape(1, 256, 256, 1)
    img = (img - 127.) / 127.
    return img


def service(model, image):
    mask = model.predict(image)
    cv2.imwrite('../images/output.png', mask[0] * 255.)
    return mask


if __name__ == '__main__':
    app.run(debug=True)