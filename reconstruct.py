import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, rotate
from skimage.data import shepp_logan_phantom
from PIL import Image
import concurrent.futures

dtheta = 2
max_angle = 180

def generate_image():
    square = rotate(np.ones([200, 200]), 30, resize=True)
    image = np.pad(square, pad_width=50)
    _ = np.linspace(-1, 1, image.shape[0])
    xv, yv = np.meshgrid(_,_)
    image[(xv-0.4)**2+(yv-0.2)**2<0.01] = 2
    return image

def forward_project(image, start_angle=0, end_angle=None):
    end_angle = end_angle or max_angle
    projections = np.array([rotate(image, -i * dtheta).sum(axis=0) for i in range(start_angle, end_angle)])
    return projections

def back_project(sinogram, start_angle=0):
    """
    Inverse Radon Transform (unfiltered)
    """
    num_angles, W = sinogram.shape
    laminogram = np.zeros((W, W))
    for i in range(num_angles):
        temp = np.tile(sinogram[i],(W,1))
        temp = rotate(temp, dtheta*(i+start_angle))
        laminogram += temp
    return laminogram

def mlem(projections, iterations=10, recon=None):
    """
    Perform Maximum Likelihood Expectation Maximization (MLEM) reconstruction.
    
    Args:
        projections: Measured projection data.
        iterations: Number of MLEM iterations to perform.
        recon: Initial reconstruction (usually a uniform image).
    Returns:
        Reconstructed image after specified number of MLEM iterations.
    """
    if recon is None:
            # Initialize the reconstruction with a uniform image if not provided
            # The shape of the reconstruction is assumed to be square with a size equal to the number of detectors
            recon = np.ones((projections.shape[1], projections.shape[1]))

    for i in range(iterations):
        # Forward project the current reconstruction
        estimated_projections = forward_project(recon)
        
        # # Prevent division by zero
        # estimated_projections[estimated_projections == 0] = 1e-10
        
        # Update the reconstruction using the ratio of measured to estimated projections
        update_ratio = projections / estimated_projections
        recon *= back_project(update_ratio)

    return recon

def osem(projections, num_subsets, iterations=10, recon=None, csize=False):
    """
    Perform Ordered Subsets Expectation Maximization (OSEM) reconstruction.
    
    Args:
        projections: Measured projection data.
        num_subsets: Number of subsets to divide the projection data into.
        iterations: Number of OSEM iterations to perform.
        recon: Initial reconstruction (usually a uniform image).
        
    Returns:
        Reconstructed image after specified number of OSEM iterations.
    """
    num_angles, W = projections.shape

    if recon is None:
        # Initialize the reconstruction with a uniform image if not provided
        recon = np.ones((W, W))
    
    if csize:
        pass
    else:
        angles = np.linspace(0, num_angles, num_subsets+1, dtype=int)
    
    for _ in range(iterations):
        for j in range(num_subsets):
            # Select the subset of projections
            start_angle, end_angle = angles[j], angles[j+1]
            subset_projections = projections[start_angle:end_angle]
            # Forward project the current reconstruction for the subset
            estimated_subset_projections = forward_project(recon, start_angle, end_angle)
            # Prevent division by zero
            estimated_subset_projections[estimated_subset_projections == 0] = 1e-10
            # Update the reconstruction using the ratio of measured to estimated projections for the subset
            update_ratio = subset_projections / estimated_subset_projections
            recon *= back_project(update_ratio, start_angle)

    return recon

import numpy as np
import concurrent.futures

def osem_parallel(projections, num_subsets, iterations=10, recon=None):
    """
    Perform parallelized Ordered Subsets Expectation Maximization (OSEM) reconstruction using ProcessPoolExecutor.
    
    Args:
        projections: Measured projection data.
        num_subsets: Number of subsets to divide the projection data into.
        iterations: Number of OSEM iterations to perform.
        recon: Initial reconstruction (usually a uniform image).
        
    Returns:
        Reconstructed image after specified number of OSEM iterations.
    """
    num_angles, W = projections.shape

    if recon is None:
        # Initialize the reconstruction with a uniform image if not provided
        recon = np.zeros((W, W))

    angles = np.linspace(0, num_angles, num_subsets+1, dtype=int)

    def process_subset(j):
        """Process a single subset of projections."""
        # Select the subset of projections
        start_angle, end_angle = angles[j], angles[j+1]
        subset_projections = projections[start_angle:end_angle]
 
        # Forward project the current reconstruction for the subset
        estimated_subset_projections = forward_project(recon, start_angle, end_angle)
        # Prevent division by zero
        estimated_subset_projections[estimated_subset_projections == 0] = 1e-10
        # Update the reconstruction using the ratio of measured to estimated projections for the subset
        update_ratio = subset_projections / estimated_subset_projections
        update_image = back_project(update_ratio, start_angle)
        return update_image

    for i in range(iterations):
        # Process each subset in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(process_subset, j) for j in range(num_subsets)]
            for f in concurrent.futures.as_completed(results):
                recon *= f.result() / num_subsets
            # for update in updates:
            #     recon *= update
            #     recon *= np.max(recon)

    return recon


def plot_bw(image):
    plt.figure(figsize=(10, 7))
    plt.imshow(image, cmap='gray')
    plt.colorbar(cmap='gray')
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
image = Image.open("scan.png")
image = image.convert('L')
print(image.format, image.size, image.mode)
image = np.asarray(image)
print(image)
# image /= np.max(image)
# image = generate_image()
# sinogram = radon(image)
# plot_bw(image)
sinogram = forward_project(image)
# sinogram /= np.max(sinogram)
plot_bw(sinogram)
# image = mlem(sinogram)
# plot_bw(image)


# plot_bw(sinogram)
# plot_bw(radon(image))
# plot_bw(back_project(sinogram))
# plot_bw(iradon(sinogram.T))
# animate_bp(sinogram.T)
