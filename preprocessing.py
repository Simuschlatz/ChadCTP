import os
import numpy as np
from pydicom import dcmread
from icecream import ic
from skimage import morphology
from scipy import ndimage
from skimage.measure import label, find_contours
import pickle
from numba import jit, prange
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor
from time import time

dataset_path = os.path.expanduser('~/Desktop/UniToBrain')

def time_benchmark(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        name = func.__name__
        print(f"{name} took {time() - t0} seconds")
        return res
    return wrapper

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
def apply_window(scan, window_center, window_width):
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



def get_2d_mask(slice: np.ndarray, threshold_min=-25, threshold_max=150, 
                remove_small_objects_size=500, structuring_element_dims=(1, 7)):
    # Apply gaussian blur
    slice = ndimage.gaussian_filter(slice, sigma=1)
    mask = get_threshold_mask(slice, min_val=threshold_min, max_val=threshold_max)
    mask = ndimage.morphology.binary_erosion(mask, np.ones(structuring_element_dims))
    # mask = ndimage.morphology.binary_erosion(mask, np.ones((3, 3)))
    # Remove small objects
    mask = morphology.remove_small_objects(mask, remove_small_objects_size)
    # Fill small holes
    # mask = ndimage.morphology.binary_dilation(mask, np.ones((7, 7)))
    # mask = ndimage.morphology.binary_fill_holes(mask)
    return mask

def get_largest_connected_component(all_masks, 
                                    morphology_shape_3d=(3, 3, 3),
                                    connectivity_shape=(3, 3, 3),
                                    morphology_shape_2d=(1, 7),
                                    ):
    # Remove 3d connections between non-brain components (eyes) and brain
    all_masks = ndimage.binary_erosion(all_masks, structure=np.ones(morphology_shape_3d))
    
    labels, num_volumes = ndimage.label(all_masks, structure=np.ones(connectivity_shape))
    # Largest connected is found in 3d, not 2d. This is because some lower slices of the brain contain multiple
    # unconnected brain components that would be removed if the largest connected component was found in 2d
    label_count = np.bincount(labels.ravel().astype(int))
    ic(label_count)
    # Exclude background
    label_count[0] = 0
    largest_volume_mask = labels == label_count.argmax()
    for i, slice_mask in enumerate(largest_volume_mask):
        # Fill small holes, improve mask in 2d
        slice_mask = ndimage.morphology.binary_dilation(slice_mask, np.ones(morphology_shape_2d))
        slice_mask = ndimage.morphology.binary_fill_holes(slice_mask)
        largest_volume_mask[i] = slice_mask
    # Account for 3D erosion
    largest_volume_mask = ndimage.morphology.binary_dilation(largest_volume_mask, np.ones(connectivity_shape))
    # for i, slice_mask in enumerate(largest_volume_mask):
    #     largest_volume_mask[i] = ndimage.morphology.binary_dilation(slice_mask, np.ones((3, 3)))
    return largest_volume_mask

@time_benchmark
def get_3d_mask(volume: np.ndarray, 
                threshold_min=-25, threshold_max=150, 
                remove_small_objects_size=500, 
                morphology_shape_2d=(1, 7),
                morphology_shape_3d=(3, 3, 3),
                connectivity_shape_3d=(3, 3, 3)
                ):
    """
    Parameters:
    - volume (numpy.ndarray): The 3D input volume to mask.
    - threshold_min (int, optional): Minimum intensity value for thresholding. Defaults to -25.
    - threshold_max (int, optional): Maximum intensity value for thresholding. Defaults to 150.
    - remove_small_objects_size (int, optional): Minimum size of objects to keep. Defaults to 500.
    - morphology_shape_2d (tuple, optional): Shape of the structuring element for 2D morphology. Defaults to (1, 7).
    - morphology_shape_3d (tuple, optional): Shape of the structuring element for 3D morphology. Defaults to (3, 3, 3).
    - connectivity_shape_3d (tuple, optional): Shape of the structuring element for 3D connectivity. Defaults to (3, 3, 3).

    Returns:
    - numpy.ndarray: A 3D binary mask of the brain.
    """
    downsampling_factor = 512 // volume.shape[1]
    if not morphology_shape_2d[1] // downsampling_factor:
        print(f"Warning: Heavy downsampling detected ({downsampling_factor}x), morphology shape will be set to (1, 1)")
        morphology_shape_2d = (1, 1)
    elif downsampling_factor > 1:
        morphology_shape_2d = (1, max(2, morphology_shape_2d[1] // downsampling_factor))
        print(f"Info: Downsampling detected ({downsampling_factor}x), adjusting morphology_shape_2d from {morphology_shape_2d} to {(1, max(2, morphology_shape_2d[1] // downsampling_factor))}")
        print(f"Info: Downsampling detected ({downsampling_factor}x), adjusting remove_small_objects_size from {remove_small_objects_size} to {remove_small_objects_size // downsampling_factor**2}")
        remove_small_objects_size = remove_small_objects_size // downsampling_factor**2
    volume_mask = np.array([get_2d_mask(s, 
                                        threshold_min=threshold_min, threshold_max=threshold_max, 
                                        remove_small_objects_size=remove_small_objects_size, 
                                        structuring_element_dims=morphology_shape_2d) 
                            for s in volume])
    return get_largest_connected_component(volume_mask, 
                                           morphology_shape_3d=morphology_shape_3d,
                                           connectivity_shape=connectivity_shape_3d,
                                           morphology_shape_2d=morphology_shape_2d
                                           )
@jit(nopython=True)
def apply_mask(a: np.ndarray, mask: np.ndarray):
    return (a + 1024) * mask - 1024

@jit(nopython=True)
def downsample(image: np.ndarray, factor=4):
    """
    Downsamples an image by a given factor
    """
    return image[::factor, ::factor]

@jit(nopython=True)
def gaussian(x_square, sigma):
    return np.exp(-0.5*x_square/sigma**2)

def bilateral_filter(image, sigma_space, sigma_intensity):
    """
    Vectorized bilateral filter implementation.
    """
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

def get_skull_mask(volume: np.ndarray, threshold: int = 700) -> np.ndarray:
    """
    Creates a binary mask of the skull using thresholding and morphological operations.
    
    Parameters:
    - volume (np.ndarray): Input CT volume in Hounsfield units
    - threshold (int): HU threshold for skull segmentation (typically ~700 HU)
    
    Returns:
    - np.ndarray: Binary mask of the skull
    """
    # Create initial skull mask
    skull_mask = volume > threshold
    
    # Perform erosion followed by dilation (opening operation)
    # This helps remove small noise while preserving skull structure
    struct_element = np.ones((3, 3, 3))
    skull_mask = ndimage.binary_erosion(skull_mask, structure=struct_element)
    skull_mask = ndimage.binary_dilation(skull_mask, structure=struct_element)
    
    return skull_mask

@time_benchmark
def register_volume(target_volume: np.ndarray, reference_image: sitk.Image) -> np.ndarray:
    """
    Registers a single volume to the reference image using rigid registration.
    Uses smoothed volumes for registration but applies transformation to original volume.
    """
    # Create smoothed versions for registration
    smoothed_target = ndimage.gaussian_filter(target_volume, sigma=1.0)  # Reduced sigma
    smoothed_reference = ndimage.gaussian_filter(
        sitk.GetArrayFromImage(reference_image), 
        sigma=1.0
    )
    
    # Convert to SimpleITK images and set physical spacing
    target_image = sitk.GetImageFromArray(smoothed_target)
    target_image.SetSpacing([1.0, 1.0, 1.0])
    reference_image.SetSpacing([1.0, 1.0, 1.0])
    
    # Create skull masks using smoothed volumes
    target_skull_mask = get_skull_mask(smoothed_target)
    reference_skull_mask = get_skull_mask(smoothed_reference)
    
    # Convert masked smoothed volumes to SimpleITK images
    target_skull_image = sitk.GetImageFromArray(smoothed_target * target_skull_mask)
    reference_skull_image = sitk.GetImageFromArray(smoothed_reference * reference_skull_mask)
    
    # Set physical properties for registration
    target_skull_image.SetSpacing([1.0, 1.0, 1.0])
    reference_skull_image.SetSpacing([1.0, 1.0, 1.0])
    
    # Initialize the transform with center of rotation at image center
    initial_transform = sitk.CenteredTransformInitializer(
        reference_skull_image, 
        target_skull_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Set up the registration method with more conservative parameters
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1,        # Increased learning rate
        minStep=1e-4,
        numberOfIterations=200,  # Reduced iterations
        gradientMagnitudeTolerance=1e-8
    )
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Add optimizer observer to prevent large transformations
    def command_iteration(method):
        if method.GetOptimizerIteration() == 0:
            print(f"Starting registration...")
        if method.GetOptimizerIteration() % 50 == 0:
            print(f"{method.GetOptimizerIteration()}: {method.GetMetricValue()}")
            # Get current transform parameters
            transform = method.GetOptimizerPosition()
            # Check for unreasonable transformations (e.g., large rotations)
            if any(abs(angle) > np.pi/4 for angle in transform[:3]):  # limit rotations to 45 degrees
                print("Warning: Large rotation detected")
                method.StopOptimization()
    
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    # Execute the registration using smoothed skull masks
    try:
        final_transform = registration_method.Execute(reference_skull_image, target_skull_image)
    except RuntimeError as e:
        print(f"Registration failed: {e}")
        return target_volume  # Return original volume if registration fails
    
    # Apply the transform to the original unsmoothed target image
    original_target_image = sitk.GetImageFromArray(target_volume)
    original_target_image.SetSpacing([1.0, 1.0, 1.0])
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)  # Standard air HU value
    
    try:
        registered_image = resampler.Execute(original_target_image)
        registered_volume = sitk.GetArrayFromImage(registered_image)
        
        # Verify the registration result
        if np.sum(registered_volume > -1000) < 0.5 * np.sum(target_volume > -1000):
            print("Warning: Registration may have failed (large portion of volume missing)")
            return target_volume
            
        ic(f"Volume registered with final metric value: {registration_method.GetMetricValue()}")
        return registered_volume
    except RuntimeError as e:
        print(f"Resampling failed: {e}")
        return target_volume

def rigid_register_volume_sequence(volume_seq: np.ndarray, reference_index: int = 1) -> np.ndarray:
    """
    Performs skull-based rigid registration on a sequence of volumes using the specified reference volume.

    Parameters:
    - volume_seq (np.ndarray): 4D array of shape (T, Y, Z, X) representing the volume sequence.
    - reference_index (int): Index of the reference volume in the sequence to which other volumes will be registered.

    Returns:
    - np.ndarray: 4D array of registered volumes with the same shape as the input.
    """
    if not (0 <= reference_index < volume_seq.shape[0]):
        raise IndexError(f"Reference index {reference_index} is out of bounds for volume sequence with length {volume_seq.shape[0]}.")

    reference_volume = volume_seq[reference_index]
    reference_image = sitk.GetImageFromArray(reference_volume)
    reference_image.SetSpacing([1.0, 1.0, 1.0])

    print(f"Starting skull-based rigid registration of {volume_seq.shape[0]} volumes to reference index {reference_index}")

    # Initialize the output array
    registered_seq = np.empty_like(volume_seq)
    registered_seq[reference_index] = volume_seq[reference_index]

    # Register each volume sequentially
    for i in range(volume_seq.shape[0]):
        if i != reference_index:
            ic(f"Registering volume {i}")
            registered_seq[i] = register_volume(volume_seq[i], reference_image)

    print("All registrations completed.")
    ic(registered_seq.shape, registered_seq.dtype)
    return registered_seq

def get_volume(folder_path, 
               windowing=True, 
               windowing_type='brain',
               extract_brain=True, 
               correct_motion=True,
               spatial_downsampling_factor=4, 
               temporal_downsampling_factor=1) -> np.ndarray:
    
    print(f"Loading {folder_path}...")
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
    print(f"Processing {folder_path}...")
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
            slice = convert_to_HU(slice, *get_conversion_params(ds))
            # slice = bilateral_filter(slice, 10, 10)
            if spatial_downsampling_factor > 1:
                slice = downsample(slice, factor=spatial_downsampling_factor)
            volume_seq[t, y] = slice

    if correct_motion:
        volume_seq = rigid_register_volume_sequence(volume_seq)
        ic(volume_seq.max(), volume_seq.min(), volume_seq.dtype)
    # Brain extraction parameters tuned for volume seq with slice size 512x512
    # So downsample after brain extraction


    if extract_brain:
        volume = volume_seq[0]
        volume_mask = get_3d_mask(volume)
        # All unmasked areas are set to -1024 HU
        volume_seq = apply_mask(volume_seq, volume_mask)
        ic(volume_seq.max(), volume_seq.min(), volume_seq.dtype)
    
            
    if windowing:
        window_center, window_width = get_window_from_type(windowing_type)
        ic(window_center, window_width)
        volume_seq = apply_window(volume_seq, window_center, window_width)
        ic(volume_seq.max(), volume_seq.min(), volume_seq.dtype)

    print(f"Done!")
    return volume_seq
    # 4. Standardization
    return (volume_seq - np.mean(volume_seq)) / np.std(volume_seq)
    # Normalization
    # return normalize(volume_seq)

def save_volume(volume, folder_path='volume.npy'):
    np.save(folder_path, volume)
    ic(f"Volume saved to {folder_path}")
