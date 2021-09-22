import requests
import streamlit as st
from visualization import *
from stream_process import *

# http://127.0.01:5000/ is from the flask api
server_url = "http://127.0.01:5000/"
process_url = "http://127.0.01:5000/process"
decoder_url = "http://127.0.01:5000/decoder"

response = requests.post(server_url)
process_response = requests.post(process_url)
decoder_response = requests.post(decoder_url)
output_ready = False

st.markdown('''
            # Lung Segmentation App
            Select a model, then upload your **CXR** image and 
            choose a pre-process for your input image
            then hit the `submit` button to get your **segmented** mask. 
            checkout `:5000` for **Flask server**.    
            For **VAE** just select to numbers between `1` and `30`, 
            hit the `submit` button and get your **generated** image.
            
            ------------------------------------------------------
            ''')

def process(file, pre_process, url: str):

    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    image = load_data(file, pre_process=pre_process)
    _, img_encoded = cv2.imencode('.jpg', image)

    r = requests.post(url,
                      data=img_encoded.tostring(),
                      headers=headers,
                      timeout=5000)

    return r


def decoder_process(vae_range, output_number, url: str):

    requested = {'vae_range': vae_range, 'output_number': output_number}
    r = requests.post(url, json=requested, timeout=5000)

    return r


with st.form(key='segmentation'):
    with st.sidebar:
        model_name = st.sidebar.selectbox(
            'Select model',
            [None, "U-Net", "Residual U-Net", "Autoencoder (VAE)"])

        if (model_name == "U-Net") or (model_name == "Residual U-Net"):
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is not None:
                pre_process = st.sidebar.radio("Pre-Process", ["Original", "DHE"])
                if file is not None:

                    processed_image = stream_data(file, pre_process=pre_process)
                    response = requests.post(server_url + model_name, timeout=5000)

                    st.write(response.content.decode("utf-8"))
                    st.image(processed_image, use_column_width=True)
                    submit_button = st.form_submit_button(label='Submit')

                    if submit_button:
                        responsed = process(file, pre_process, process_url)
                        if responsed.status_code == 200:
                            output_ready = True

        elif model_name == "Autoencoder (VAE)":
            vae_range = st.sidebar.slider("Autoencoder range", 0, 30, step=1)
            output_number = st.sidebar.slider("How many image?", 0, 30, step=1)

            submit_button = st.form_submit_button(label='Submit')
            if submit_button:
                responsed = decoder_process(vae_range, output_number, decoder_url)
                if responsed.status_code == 200:
                    output_ready = True


if output_ready:
    if model_name == "Autoencoder (VAE)":
        st.write("""
                    ### {} iamges in range [-{}, +{}]
                    """.format(output_number ** 2, vae_range, vae_range))
        st.image('images/output_vae.png', use_column_width=True)
        st.success('images successfully generated! :thumbsup:')

    else:
        st.write("""
                    ### input CXR and detected lung
                    """)
        visualize_output(processed_image, 'images/output.png')
        st.image('images/output.png')
        st.success('Mask detected successfully! :thumbsup:')