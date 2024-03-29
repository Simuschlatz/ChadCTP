import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, rotate
from skimage.data import shepp_logan_phantom
from time import sleep
dtheta = 1

def generate_image():
    square = rotate(np.ones([200, 200]), 30, resize=True)
    image = np.pad(square, pad_width=50)
    _ = np.linspace(-1, 1, image.shape[0])
    xv, yv = np.meshgrid(_,_)
    image[(xv-0.1)**2+(yv-0.2)**2<0.01] = 2
    return image

def get_sinogram(image):
    thetas = np.arange(0, 180, dtheta)
    projections = np.array([rotate(image, theta).sum(axis=1) for theta in thetas])
    return projections.T

def back_project(sinogram):
    """
    Inverse Radon Transform (unfiltered)
    """
    laminogram = np.zeros((sinogram.shape[1], sinogram.shape[1]))
    for i in range(sinogram.shape[0]):
        temp = np.tile(sinogram[i],(sinogram.shape[1],1))
        temp = rotate(temp, dtheta*i)
        laminogram += temp
    return laminogram

def plot_bw(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def animate_bp(p):
    plt.ion()
    laminogram = np.zeros((sinogram.shape[1], sinogram.shape[1]))
    # fig, ax = plt.subplots()
    window = plt.imshow(laminogram, cmap='gray')
    # ax.imshow(laminogram)
    for i in range(sinogram.shape[0]):
        # sleep(.5)
        print(i)
        temp = np.tile(sinogram[i],(sinogram.shape[1],1))
        temp = rotate(temp, dtheta*i)
        laminogram += temp
        # window.imshow(laminogram, cmap='gray')
        # fig.canvas.draw_idle()
        window.set_data(laminogram)
        plt.show()
        plt.pause(0.1)
    # plt.show()
    return laminogram

image = shepp_logan_phantom()
sinogram = radon(image)
# plot_bw(generate_image())
# plot_bw(sinogram)
plot_bw(back_project(sinogram.T))
# plot_bw(iradon(sinogram))
# animate_bp(sinogram.T)