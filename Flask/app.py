import cv2
import jsonpickle
from unet import *
from resunet import *
from vae import decoder
from visualization import request_handler, service, visualize_vae
from flask import Flask, request, Response, render_template

model_unet = unet(pretrained_weights='../weigths/cxr_seg_unet.hdf5')
model_runet = ResUNet(pretrained_weights='../weigths/cxr_seg_res_unet.hdf5')
model_decoder = decoder(pretrained_weights='../weigths/decoder.hdf5')

app = Flask(__name__)

# home page for Flask
@app.route('/')
def hello():
    return render_template('index.html')


# Network selected
@app.route('/<name>', methods=['GET', 'POST'])
def name(name):
    global selected_model
    selected_model = name
    return f"you're using {name}!\n\nHere's your input image:"


# Process unit
@app.route('/process', methods=['POST'])
def get_segmented():

    image = request_handler(request)
    if selected_model == "U-Net":
        service(model_unet, image)
    elif selected_model == "Residual U-Net":
        service(model_runet, image)

    response = {'message': "Mask saved successfully!"}

    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


@app.route('/decoder', methods=['POST'])
def vae():
    r = request.json
    vae_range, output_number = r['vae_range'], r['output_number']
    if (vae_range != 0) and (output_number != 0):
        fig = visualize_vae(model_decoder, output_number, vae_range)
        cv2.imwrite('../images/output_vae.png', fig * 255.)

    response = {'message': "decoded image saved successfully!"}

    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(debug=True)