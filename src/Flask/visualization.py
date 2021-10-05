import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
    get inputed CXR image and path to created_mask and attached them together as PNG.  
"""
def visualize_output(processed_image, output_image_path):
    output_figure = np.zeros((processed_image.shape[1] * processed_image.shape[0], processed_image.shape[2] * 2, 1))
    mask = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE).reshape(256, 256, 1)

    output_figure[0: 256, 0: 256] = (processed_image[0] * 255.)
    output_figure[0: 256, 256: 512] = mask

    fig_shape = np.shape(output_figure)
    output_figure = output_figure.reshape((fig_shape[0], fig_shape[1]))
    cv2.imwrite('../../images/output.png', output_figure)


"""
    attached created images from VAE save as PNG.
"""
def visualize_vae(decoder, output_number, vae_range):
    dim = 256
    figure = np.zeros((dim * output_number, dim * output_number, 1))

    grid_x = np.linspace(-vae_range, vae_range, output_number)
    grid_y = np.linspace(-vae_range, vae_range, output_number)[::-1]

    # decoder for each square in the grid
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(dim, dim, 1)
            figure[i * dim: (i + 1) * dim,
            j * dim: (j + 1) * dim] = digit

    plt.figure(figsize=(10, 10))
    # Reshape for visualization
    fig_shape = np.shape(figure)
    figure = figure.reshape((fig_shape[0], fig_shape[1]))
    # cv2.imwrite('images/output_vae.png', figure * 255.)
    return figure


"""
    Handle all requests from frontend (Streamlit) to backend (FLask)
"""
def request_handler(request):
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = img.reshape(1, 256, 256, 1)
    img = (img - 127.) / 127.
    return img

"""
    here machine predict mask for CXR image
"""
def service(model, image):
    mask = model.predict(image)
    cv2.imwrite('../../images/output.png', mask[0] * 255.)
