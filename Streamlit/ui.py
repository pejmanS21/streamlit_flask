# app.py
import streamlit as st
import requests
from stream_process import *
from requests_toolbelt.multipart.encoder import MultipartEncoder

# http://127.0.01:5000/ is from the flask api

server_url = "http://127.0.01:5000/"
process_url = "http://127.0.01:5000/process"

response = requests.post(server_url)
process_response = requests.post(process_url)

def process(file, pre_process, url: str):
    """
    img = cv2.imread('lena.jpg')
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    """
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    image = load_data(file, pre_process=pre_process)
    _, img_encoded = cv2.imencode('.jpg', image)

    r = requests.post(url,
                      data=img_encoded.tostring(),
                      headers=headers,
                      timeout=5000)

    return r


with st.form(key='segmentation'):
    with st.sidebar:
        model_name = st.sidebar.selectbox(
            'Select model',
            [None, "U-Net", "Residual U-Net", "Autoencoder (VAE)"])

        if model_name == "U-Net":
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is not None:
                pre_process = st.sidebar.radio("Pre-Process", ["Original", "DHE"])
                if (file is not None) and (model_name == "U-Net"):

                    processed_image = stream_data(file, pre_process=pre_process)
                    response = requests.post(server_url + model_name, timeout=5000)

                    st.write(response.content.decode("utf-8"))
                    st.image(processed_image, use_column_width=True)
                    submit_button = st.form_submit_button(label='Submit')

                    if submit_button:
                        responsed = process(file, pre_process, process_url)
                        print(responsed)
                        print(responsed.content)
                        st.image(responsed, use_column_width=True)