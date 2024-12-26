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
from registration import register_volume_inplane_weighted

dataset_path = os.path.expanduser('~/Desktop/UniToBrain')
num_samples = 10
def time_benchmark(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        name = func.__name__
        print(f"{name} took {time() - t0} seconds")
        return res
    return wrapper

def save_folder_paths(num_slices: int=288, output_file: str='folder_paths.pkl'):
    folder_paths = []
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if len(folder) == 7 and len(os.listdir(folder_path)) == num_slices: # MOL-XYZ
            folder_paths.append(folder_path)
    print(f"{len(folder_paths)} folders found")

    output_file = os.path.join(dataset_path, output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(folder_paths, f)

    print(f"Folder paths saved to {output_file}")
    return folder_paths

def load_folder_paths(scan_size='small'):
    """
    ``scan_size``: 'small' or 'large'. If 'small', only load folders with 288 slices 
    (18 volumes with 16 slices each). If 'large' only load folders with 712 slices 
    (89 volumes with 8 slices each).
    """
    num_slices = 288 if scan_size == 'small' else 712
    output_file = f"folder_paths_{num_slices}.pkl"
    file_path = os.path.join(dataset_path, output_file)
    if not os.path.exists(file_path):
        print(f"File {output_file} does not exist, running save_folder_paths() instead...")
        return save_folder_paths(num_slices=num_slices, output_file=output_file)
    
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
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} not found")
    
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')])
    print(f"Dicom files loaded, count: {len(files)}")

    return [dcmread(f) for f in files] # Pydicom datasets

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
    """
    Performs brain extraction on a 2D slice using thresholding, morphological operations, 
    and connected component analysis.
    """
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
    """
    Finds the largest connected component in a 3D volume using 3D connectivity analysis.
    2D and 3D morphological operations and as well as connected component analysis are performed to 
    improve the mask quality.

    Returns:
    - numpy.ndarray: A 3D binary mask of the largest connected component, ideally the brain.
    """
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
    # Account for 3D erosion
    largest_volume_mask = ndimage.morphology.binary_dilation(largest_volume_mask, np.ones(connectivity_shape))
    for i, slice_mask in enumerate(largest_volume_mask):
        # Fill small holes, improve mask in 2d
        slice_mask = ndimage.morphology.binary_dilation(slice_mask, np.ones(morphology_shape_2d))
        slice_mask = ndimage.morphology.binary_fill_holes(slice_mask)
        largest_volume_mask[i] = slice_mask
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
    Performs brain extraction on a 3D volume using thresholding, 2D and 3D morphological operations, 
    and 2D and 3D connected component analysis.

    Parameters:
    - `volume` (numpy.ndarray): The 3D input volume to mask.
    - `threshold_min` (int, optional): Minimum intensity value for thresholding. Defaults to -25.
    - `threshold_max` (int, optional): Maximum intensity value for thresholding. Defaults to 150.
    - `remove_small_objects_size` (int, optional): Minimum size of objects to keep. Defaults to 500.
    - `morphology_shape_2d` (tuple, optional): Shape of the structuring element for 2D morphology. Defaults to (1, 7).
    - `morphology_shape_3d` (tuple, optional): Shape of the structuring element for 3D morphology. Defaults to (3, 3, 3).
    - `connectivity_shape_3d` (tuple, optional): Shape of the structuring element for 3D connectivity. Defaults to (3, 3, 3).

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
def register_volume_3D(target_volume: np.ndarray, reference_image: sitk.Image) -> np.ndarray:
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
        return registered_volume#, registration_method.GetMetricValue()
    except RuntimeError as e:
        print(f"Resampling failed: {e}")
        return target_volume
    
@time_benchmark
def register_volume_2D(target_volume: np.ndarray, reference_image: sitk.Image, spacing: np.ndarray = (1.0, 1.0, 1.0)) -> np.ndarray:
    """
    Registers a single volume to the reference image using 2D rigid registration slice by slice.
    Uses smoothed volumes for registration but applies transformation to original volume.
    """
    reference_volume = sitk.GetArrayFromImage(reference_image)
    registered_volume = np.empty_like(target_volume)
    
    # Process each slice independently
    for slice_idx in range(target_volume.shape[0]):
        # Get corresponding slices
        target_slice = target_volume[slice_idx]
        reference_slice = reference_volume[slice_idx]
        
        # Create smoothed versions for registration
        smoothed_target = ndimage.gaussian_filter(target_slice, sigma=1.0)
        smoothed_reference = ndimage.gaussian_filter(reference_slice, sigma=1.0)
        
        # Convert to SimpleITK images
        target_image = sitk.GetImageFromArray(smoothed_target)
        reference_slice_image = sitk.GetImageFromArray(smoothed_reference)
        
        # Set physical properties
        target_image.SetSpacing([1.0, 1.0])
        reference_slice_image.SetSpacing([1.0, 1.0])
        
        # Initialize the 2D transform
        initial_transform = sitk.CenteredTransformInitializer(
            reference_slice_image,
            target_image,
            sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        # Set up registration method
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=0.2,
            minStep=1e-4,
            numberOfIterations=500,
            gradientMagnitudeTolerance=1e-8
        )
        registration_method.SetInitialTransform(initial_transform)
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Add optimizer observer
        def command_iteration(method):
            if method.GetOptimizerIteration() == 0:
                print(f"Starting registration for slice {slice_idx}...")
            if method.GetOptimizerIteration() % 50 == 0:
                print(f"Slice {slice_idx} - Iteration {method.GetOptimizerIteration()}: {method.GetMetricValue()}")
                transform = method.GetOptimizerPosition()
                    
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
        
        try:
            final_transform = registration_method.Execute(reference_slice_image, target_image)
            
            # Apply transform to original slice
            original_slice_image = sitk.GetImageFromArray(target_slice)
            original_slice_image.SetSpacing([1.0, 1.0])
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(reference_slice_image)
            resampler.SetTransform(final_transform)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(-1024)
            
            registered_slice = sitk.GetArrayFromImage(resampler.Execute(original_slice_image))
            
            # Verify registration result
            if np.sum(registered_slice > -1000) < 0.5 * np.sum(target_slice > -1000):
                print(f"Warning: Registration may have failed for slice {slice_idx}")
                registered_volume[slice_idx] = target_slice
            else:
                registered_volume[slice_idx] = registered_slice
                
        except RuntimeError as e:
            print(f"Registration failed for slice {slice_idx}: {e}")
            registered_volume[slice_idx] = target_slice
            
    return registered_volume
def rigid_register_volume_sequence(volume_seq: np.ndarray, 
                                   spacing: np.ndarray = (1.0, 1.0, 1.0),
                                   reference_index: int = 1,
                                   registering_structure: str = 'all',
                                   window_center: int = 40, 
                                   window_width: int = 400) -> np.ndarray:
    """
    Performs skull-based rigid registration on a sequence of volumes using the specified reference volume.

    Parameters:
    - volume_seq (np.ndarray): 4D array of shape (T, Y, Z, X) representing the volume sequence.
    - reference_index (int): Index of the reference volume in the sequence to which other volumes will be registered.
    - registering_structure (str): The structure used in registration. One of: 'all', 'skull', 'brain'. Defaults to 'all'.
    - window_center (int): The center of the window in HU. Defaults to 40.
    - window_width (int): The width of the window in HU. Defaults to 400.

    Returns:
    - np.ndarray: 4D array of registered volumes with the same shape as the input.
    """
    if not (0 <= reference_index < volume_seq.shape[0]):
        raise IndexError(f"Reference index {reference_index} is out of bounds for volume sequence with length {volume_seq.shape[0]}.")

    # Window the sequence to reduce impact of noise or outliers
    volume_seq = apply_window(volume_seq, window_center, window_width)
    reference_volume = volume_seq[reference_index]
    if registering_structure == 'all':
        reference_image = sitk.GetImageFromArray(reference_volume)
    elif registering_structure == 'skull':
        reference_image = sitk.GetImageFromArray(get_skull_mask(reference_volume))
    elif registering_structure == 'brain':
        reference_image = sitk.GetImageFromArray(get_3d_mask(reference_volume))
    else:
        raise ValueError(f"Invalid registering structure: {registering_structure}. Must be one of: 'all', 'skull', 'brain'.")
    reference_image.SetSpacing(spacing)

    # print(f"Starting skull-based rigid registration of {volume_seq.shape[0]} volumes to reference index {reference_index}")

    # Initialize the output array
    registered_seq = np.empty_like(volume_seq)
    registered_seq[reference_index] = volume_seq[reference_index]

    # Register each volume sequentially
    for i in range(volume_seq.shape[0]):
        if i == reference_index: continue
        ic(f"Registering volume {i}")
        registered_seq[i] = register_volume_inplane_weighted(volume_seq[i], reference_image, n_samples=5, weighting_scheme='inverse')

    print("All registrations completed.")
    ic(registered_seq.shape, registered_seq.dtype)
    return registered_seq

def get_volume(folder_path, 
               windowing=True, 
               windowing_type='brain',
               filter=True,
               extract_brain=True, 
               correct_motion=True,
               reference_index=1,
               spatial_downsampling_factor=4, 
               temporal_downsampling_factor=1,
               verbose=True) -> np.ndarray:
    """
    Processes the DICOM files in a folder with folder path `folder_path`.
    Each folder contains the entire perfusion volume sequence as DICOM datasets
    The objective is to convert the sequence into a 4D array of CT volumes that are
    in HU, windowed, brain-extracted, registered, filtered, standardized

    Parameters:
    - `extract_brain` (bool, optional): Whether to extract the brain from the volume sequence. Defaults to True.
    - `correct_motion` (bool, optional): Whether to correct motion in the volume sequence. Defaults to True.
    - `reference_index` (int, optional): Index of the reference volume in the sequence to which other volumes will be registered. Defaults to 1.
    - `spatial_downsampling_factor` (int, optional): Factor by which to downsample the volume sequence in the spatial dimensions. Defaults to 4.
    - `temporal_downsampling_factor` (int, optional): Factor by which to downsample the volume sequence in the temporal dimension. Defaults to 1.
    """
    if verbose: print(f"Loading {folder_path}...")
    datasets = load_dcm_datasets(folder_path)
    
    if temporal_downsampling_factor < 1:
        if verbose: print("Warning: temporal downsampling factor is less than 1, setting to 1")
        temporal_downsampling_factor = 1
    if spatial_downsampling_factor < 1:
        if verbose: print("Warning: spatial downsampling factor is less than 1, setting to 1")
        spatial_downsampling_factor = 1

    if verbose: print(f"Processing {folder_path}...")
    ds = datasets[0]

    # Get the spacing of the volume sequence
    slice_thickness = ds.SliceThickness
    pixel_spacing = ds.PixelSpacing
    spacing = np.array([slice_thickness, pixel_spacing[0], pixel_spacing[1]])
    print(slice_thickness, pixel_spacing)
    # Assume that each volume in the sequence has the same dimensions
    Y = int((datasets[-1].SliceLocation - ds.SliceLocation + 5) // ds.SliceThickness) # Height
    Z, X = ds.Rows // spatial_downsampling_factor, ds.Columns // spatial_downsampling_factor # Depth, Width
    n_volumes = len(datasets) // Y
    T = max(1, n_volumes // temporal_downsampling_factor) # Temporal dimension
    volume_seq = np.empty((T, Y, Z, X), dtype=np.float32)

    for t in range(T):
        for y in range(Y):
            slice = datasets[t * Y * temporal_downsampling_factor + y].pixel_array
            slice = convert_to_HU(slice, *get_conversion_params(ds))
            if spatial_downsampling_factor > 1:
                slice = downsample(slice, factor=spatial_downsampling_factor)
            if filter: slice = bilateral_filter(slice, 10, 10)
            volume_seq[t, y] = slice

    if correct_motion:
        volume_seq = rigid_register_volume_sequence(volume_seq, spacing=spacing, reference_index=reference_index)
        if verbose: ic(volume_seq.max(), volume_seq.min(), volume_seq.dtype)

    if extract_brain:
        volume = volume_seq[0]
        volume_mask = get_3d_mask(volume)
        # All unmasked areas are set to -1024 HU
        volume_seq = apply_mask(volume_seq, volume_mask)
        if verbose: ic(volume_seq.max(), volume_seq.min(), volume_seq.dtype)
    

    if windowing:
        window_center, window_width = get_window_from_type(windowing_type)
        ic(window_center, window_width)
        volume_seq = apply_window(volume_seq, window_center, window_width)
        if verbose: ic(volume_seq.max(), volume_seq.min(), volume_seq.dtype)

    if verbose: print(f"Done!")
    return volume_seq
    # 4. Standardization
    return (volume_seq - np.mean(volume_seq)) / np.std(volume_seq)
    # Normalization
    # return normalize(volume_seq)

def save_volume(volume, folder_path='volume.npy'):
    np.save(folder_path, volume)
    ic(f"Volume saved to {folder_path}")

if __name__ == "__main__":
    # Take all volumes with scan size 'small' --> 18x16x512x512 (T, Y, Z, X)
    for folder_path in load_folder_paths(scan_size='small')[:num_samples]:
        volume_seq = get_volume(folder_path)
        save_volume(volume_seq, folder_path.replace('.dcm', '.npy'))
