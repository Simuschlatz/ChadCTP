import os
import numpy as np
from pydicom import dcmread
from icecream import ic
from skimage import morphology
from scipy import ndimage
from skimage.measure import label, find_contours
import pickle
from numba import jit, prange
from time import time
# from cv2 import bilateral_filter
# from pydicom.pixel_data_handlers.util import apply_windowing

# Remove the class definition and convert class variables to module-level constants
windowing_lookup = {
    "lung": (-600, 1500),
    "mediastinum": (50, 350),
    "tissues": (50, 400),
    "brain": (40, 80),
    "bone": (400, 1800)
}

def load_dcm_datasets(folder_path: str) -> list:
    if folder_path.endswith('.npy'):
        return np.load(folder_path)
    try:
        files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')])
        ic(f"Dicom files loaded, count: {len(files)}")

        return [dcmread(f) for f in files] # Pydicom datasets
    except FileNotFoundError:
        raise FileNotFoundError(f"Folder {folder_path} not found")

@jit(nopython=True)
def convert_to_HU(slice, intercept, slope):
    return slice * slope + intercept

@jit(nopython=True)
def normalize(scan):
    min_val, max_val = scan.min(), scan.max()
    return (scan - min_val) / (max_val - min_val)

@jit(nopython=True)
def apply_windowing(scan, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    return np.clip(scan, img_min, img_max)

def get_window_from_type(type: str):
    try:
        return windowing_lookup[type]
    except KeyError:
        ic(f"Windowing type {type} not found in windowing lookup")
        return None

def get_windowing_params(ds):
    return ds.WindowCenter, ds.WindowWidth

def get_conversion_params(ds):
    return ds.RescaleIntercept, ds.RescaleSlope

def is_homogenous_windowing_params(datasets):
    params = get_windowing_params(datasets[0])
    return all(get_windowing_params(ds) == params for ds in datasets[1:])

@jit(nopython=True)
def threshold(ct_volume, skull_threshold=700):
    """
    Segments the brain from the CT volume using thresholding and 3D unet
    A fundamental step in neuroimage preprocessing \n
    ``skull_threshold``: Thresholding HU value to remove skull, values from 
    https://www.sciencedirect.com/topics/medicine-and-dentistry/hounsfield-scale
    """
    thresholded = np.where(ct_volume >= skull_threshold, -1000, ct_volume)
    return thresholded

@jit(nopython=True)
def get_threshold_mask(slice, min_val=-40, max_val=120):
    return (min_val < slice) & (slice < max_val)

def get_brain_mask(image: np.ndarray):
    segmentation = morphology.dilation(image, np.ones((5, 5)))
    labels, _ = ndimage.label(segmentation)
    label_count = np.bincount(labels.ravel().astype(int))
    # The size of label_count is the number of classes/segmentations found

    # We don't use the first class since it's the background
    label_count[0] = 0

    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()

    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    return mask
    

def cv2_bilateral_filter(volume_seq, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(volume_seq, d, sigmaColor, sigmaSpace)

def rigid_registration(volume_sequence):
    """
    Performs rigid registration on a sequence of CT brain volumes.
    
    This function first denoises the volumes and then applies rigid registration
    to align all volumes in the sequence with the first volume.
    
    Parameters:
    - volume_sequence (numpy.ndarray): 4D array of CT volumes (time, z, y, x)
    
    Returns:
    - numpy.ndarray: Registered 4D volume sequence
    """
    from scipy import ndimage
    from skimage.restoration import denoise_nl_means
    from skimage.transform import AffineTransform
    from skimage.registration import optical_flow_tvl1
    
    # Denoise each volume in the sequence
    denoised_sequence = np.zeros_like(volume_sequence)
    for t in range(volume_sequence.shape[0]):
        denoised_sequence[t] = denoise_nl_means(volume_sequence[t], h=0.1, fast_mode=True, patch_size=5, patch_distance=6)
    
    # Use the first volume as the reference
    reference = denoised_sequence[0]
    registered_sequence = np.zeros_like(denoised_sequence)
    registered_sequence[0] = reference
    
    # Perform rigid registration for each subsequent volume
    for t in range(1, denoised_sequence.shape[0]):
        # Estimate transformation using optical flow
        flow = optical_flow_tvl1(reference, denoised_sequence[t])
        
        # Convert flow to affine transformation
        affine = AffineTransform(translation=(-flow[0].mean(), -flow[1].mean()))
        
        # Apply transformation
        registered_sequence[t] = ndimage.affine_transform(denoised_sequence[t], affine.params)
    
    return registered_sequence

def get_mask(slice, threshold_min=-25, threshold_max=150, remove_small_objects_size=500):
    # Apply gaussian blur
    slice = ndimage.gaussian_filter(slice, sigma=1)
    mask = get_threshold_mask(slice, min_val=threshold_min, max_val=threshold_max)
    mask = ndimage.morphology.binary_erosion(mask, np.ones((5, 5)))
    mask = ndimage.morphology.binary_erosion(mask, np.ones((3, 3)))
    # Remove small objects
    mask = morphology.remove_small_objects(mask, remove_small_objects_size)
    # Fill small holes
    mask = ndimage.morphology.binary_dilation(mask, np.ones((8, 8)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    return mask

def skull_strip(slice):
    mask = get_mask(slice)
    objects = label(mask)
    slice = (slice + 1024) * mask - 1024
    # slice = apply_windowing(slice, 40, 80)
    return slice
    mask = get_brain_mask(slice)
    return slice * mask

@jit(nopython=True)
def genCont(image, cont):
    """Function to create image contour from coordinates"""
    cont_imag = np.zeros_like(image)
    for ii in range(len(cont)):
        cont_imag[cont[ii,0],cont[ii,1]] = 1
    return cont_imag


def skull_strip_mask(cls, image: np.ndarray, bone_hu=110, ct_inf=-110, ct_sup=120):
    """
    Creates a skull-stripping mask for CT images.

    This function generates a binary mask to separate the brain tissue from the skull
    and other non-brain structures in CT images. It uses a combination of thresholding,
    morphological operations, and contour analysis to identify and isolate the brain region.

    Parameters:
    - image (numpy.ndarray): The input CT image.
    - bone_hu (int): The Hounsfield Unit threshold for bone, default is 110.
    - ct_inf (int): The lower Hounsfield Unit threshold for brain tissue, default is -110.
    - ct_sup (int): The upper Hounsfield Unit threshold for brain tissue, default is 120.

    Returns:
    - tuple: A tuple containing:
        - mask (numpy.ndarray): The binary skull-stripping mask.
        - use_Tmax (int): Flag indicating if an inner portion of the brain is present (1) or not (0).

    Note:
    This method assumes that the input image is a 2D slice from a CT volume.
    The resulting mask can be applied to the original image to remove non-brain structures.
    """
    img_max = np.max(image)
    # Selecting areas with certain
    image_mask = np.zeros_like(image).astype(int)
    image_mask[(bone_hu < image) & (image < img_max)] = 1
    # Removing objects with area smaller than a certain values
    image_mask_clean = morphology.remove_small_objects(image_mask.astype(bool), 1500)

    # Improving skull definition
    labels, label_nb = ndimage.label(image_mask_clean)
    se = morphology.disk(10)
    close_small_bin = morphology.closing(labels, se)
    # Finding contours of the various areas
    contours = find_contours(close_small_bin,0)

    # Creating masks of the various rounded areas
    areas = []
    masks = []
    for contour in contours:
        cont = cls.genCont(image_mask_clean,np.array(contour,dtype=int)).astype(int)
        mask = morphology.dilation(cont, np.ones((2, 2)))
        mask = ndimage.morphology.binary_fill_holes(mask)
        mask = morphology.dilation(mask, np.ones((3, 3)))
        masks.append(mask.copy())
        # Computing areas to find correct inner portion
        areas.append(np.sum(mask.ravel()))

    #use_Tmax = flag to check if an inner portion of the brain is present in the current section
    if len(areas) == 1:
        # If only one contour is found, there is no inner portion
        mask = masks[0].astype(np.float32)
        # use_Tmax = 0
    elif len(areas) > 1:
        # If two or more contours have been found, take the second-largest one as inner portion
        sort_idx = np.argsort(areas)
        mask = masks[sort_idx[1]].astype(np.float32)
        # use_Tmax = 1

        # Improving skull definition
        maskedImg = image * mask
        image_mask = np.zeros_like(maskedImg).astype(int)
        image_mask[(ct_inf < maskedImg) & (maskedImg < ct_sup)] = 1
        # Removing objects with area smaller than a certain values
        image_mask_clean = morphology.remove_small_objects(image_mask.astype(bool), 3000)

        # Improving skull definition again
        labels, label_nb = ndimage.label(image_mask_clean)
        se = morphology.disk(2)
        close_small_bin = morphology.closing(labels, se)
        if close_small_bin.max() == 2:
            image_mask = np.zeros_like(maskedImg).astype(int)
            image_mask[close_small_bin == 2] = 1
            image_mask_clean = morphology.remove_small_objects(image_mask.astype(bool), 6000)
            mask = ndimage.morphology.binary_fill_holes(image_mask_clean)
            maskedImg = image * mask
    else:
        # No areas have been found
        mask = np.zeros_like(image_mask_clean, dtype=np.float32)
        use_Tmax = 0
    return mask#, use_Tmax

@jit(nopython=True)
def downsample(volume_seq: np.ndarray, factor=4):
    """
    Downsamples a """
    return volume_seq[::factor, ::factor]

def get_volume(folder_path, windowing=True, extract_brain=True, spatial_downsampling_factor=4, temporal_downsampling_factor=1) -> np.ndarray:
    datasets = load_dcm_datasets(folder_path)
    
    if temporal_downsampling_factor < 1:
        print("Warning: temporal downsampling factor is less than 1, setting to 1")
        temporal_downsampling_factor = 1
    if spatial_downsampling_factor < 1:
        print("Warning: spatial downsampling factor is less than 1, setting to 1")
        spatial_downsampling_factor = 1

    # Process the DICOM files
    # Each file contains the entire perfusion volume sequence as DICOM datasets
    # The objective is to convert the sequence into a 4D array of CT volumes that are
    # in HU, windowed, brain-extracted, normalized, registered and filtered

    ds = datasets[0]
    # print(ds)
    # Assume that each volume in the sequence has the same dimensions
    Y = int((datasets[-1].SliceLocation - datasets[0].SliceLocation + 5) // ds.SliceThickness) # Height
    Z, X = ds.Rows // spatial_downsampling_factor, ds.Columns // spatial_downsampling_factor # Depth, Width
    n_volumes = len(datasets) // Y
    T = n_volumes // temporal_downsampling_factor # Temporal dimension
    volume_seq = np.empty((T, Y, Z, X), dtype=np.float32)

    for t in range(T):
        for y in range(Y):
            slice = datasets[t * Y * temporal_downsampling_factor + y].pixel_array
            if spatial_downsampling_factor > 1:
                slice = downsample(slice, factor=spatial_downsampling_factor)
            slice = convert_to_HU(slice, *get_conversion_params(ds))
            slice = bilateral_filter(slice, 2.5, 5)
            if extract_brain:
                # pass
                slice = skull_strip(slice)
            volume_seq[t, y] = slice

    # volume_seq = rigid_registration(volume_seq)

    if windowing:
        window_center, window_width = get_window_from_type('brain')
        ic(window_center, window_width)
        volume_seq = apply_windowing(volume_seq, window_center, window_width)

    return volume_seq
    # 4. Standardization
    return (volume_seq - np.mean(volume_seq)) / np.std(volume_seq)
    # Normalization
    # return normalize(volume_seq)

def save_volume(volume, folder_path='volume.npy'):
    np.save(folder_path, volume)
    ic(f"Volume saved to {folder_path}")

@jit(nopython=True, parallel=True)
def bilateral_3d_filter(volume_seq, sigma_space, sigma_intensity):
    def gaussian(x_square, sigma):
        return np.exp(-0.5 * x_square / sigma**2)
    
    time, depth, height, width = volume_seq.shape
    result = np.zeros_like(volume_seq)
    
    print("Pre-computing spatial kernel...")
    # Pre-compute spatial kernel
    kernel_size = int(2 * sigma_space + 1)
    spatial_kernel = np.zeros((kernel_size, kernel_size, kernel_size))
    for z in range(kernel_size):
        for y in range(kernel_size):
            for x in range(kernel_size):
                spatial_kernel[z, y, x] = gaussian(
                    (z - kernel_size // 2)**2 + 
                    (y - kernel_size // 2)**2 + 
                    (x - kernel_size // 2)**2, 
                    sigma_space
                )
    
    print("Applying bilateral filter...")
    for t in prange(time):
        for k in range(depth):
            for i in range(height):
                print(f"Processing slice {i} of {height} of time {t}")
                for j in range(width):
                    total_weight = 0
                    filtered_value = 0
                    
                    for z in range(max(0, k - kernel_size // 2), min(depth, k + kernel_size // 2 + 1)):
                        for y in range(max(0, i - kernel_size // 2), min(height, i + kernel_size // 2 + 1)):
                            for x in range(max(0, j - kernel_size // 2), min(width, j + kernel_size // 2 + 1)):
                                spatial_weight = spatial_kernel[
                                    z - (k - kernel_size // 2),
                                    y - (i - kernel_size // 2),
                                    x - (j - kernel_size // 2)
                                ]
                                intensity_diff = volume_seq[t, k, i, j] - volume_seq[t, z, y, x]
                                intensity_weight = gaussian(intensity_diff**2, sigma_intensity)
                                weight = spatial_weight * intensity_weight
                                
                                filtered_value += weight * volume_seq[t, z, y, x]
                                total_weight += weight
                    
                    result[t, k, i, j] = filtered_value / total_weight
    
    return result

@jit(nopython=True)
def gaussian(x_square, sigma):
    return np.exp(-0.5*x_square/sigma**2)

def bilateral_filter(image, sigma_space, sigma_intensity):

    # kernel_size should be twice the sigma space to avoid calculating negligible values
    kernel_size = int(2*sigma_space)
    half_kernel_size = kernel_size // 2
    result = np.zeros(image.shape)
    W = 0

    # Iterating over the kernel
    for x in range(-half_kernel_size, half_kernel_size+1):
        for y in range(-half_kernel_size, half_kernel_size+1):
            # Spatial Gaussian
            Gspace = gaussian(x ** 2 + y ** 2, sigma_space)
            # Shifted image
            shifted_image = np.roll(image, [x, y], [1, 0])
            # Intensity difference image
            intensity_difference_image = image - shifted_image
            # Intensity Gaussian
            Gintenisity = gaussian(
                intensity_difference_image ** 2, sigma_intensity)
            # Weighted sum
            result += Gspace*Gintenisity*shifted_image
            # Total weight
            W += Gspace*Gintenisity

    return result / W

def filter_volume_seq(volume_seq, sigma_space, sigma_intensity):
    total_time = 0
    for i, volume in enumerate(volume_seq):
        print(f"Filtering volume {i} of {len(volume_seq)}")
        for j, slice in enumerate(volume):
            # print(f"Filtering slice {j} of {len(volume)} of volume {i} of {len(volume_seq)}")
            start_time = time()
            volume_seq[i, j] = bilateral_filter(slice, sigma_space, sigma_intensity)
            end_time = time()
            total_time += end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time taken per slice: {total_time / (len(volume_seq) * len(volume_seq[0])):.2f} seconds")
    # return volume_seq

dataset_path = os.path.expanduser('~/Desktop/UniToBrain')

def save_folder_paths(output_file: str='folder_paths.pkl'):
    folder_paths = []
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if len(folder) == 7 and len(os.listdir(folder_path)) != 288: # MOL-XYZ
            folder_paths.append(folder_path)
    ic(len(folder_paths), folder_paths[len(folder_paths)-10:])

    output_file = os.path.join(dataset_path, output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(folder_paths, f)

    ic(f"Folder paths saved to {output_file}")
    return folder_paths

def load_folder_paths(output_file: str='folder_paths_288.pkl'):
    file_path = os.path.join(dataset_path, output_file)
    if not os.path.exists(file_path):
        print(f"File {output_file} does not exist, running save_folder_paths() instead...")
        return save_folder_paths(output_file=output_file)
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)