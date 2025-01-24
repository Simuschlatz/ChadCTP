import os
import numpy as np
from pydicom import dcmread
from icecream import ic
from skimage import morphology
from scipy import ndimage
# from skimage.measure import label, find_contours
import pickle
from numba import jit
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
from time import time

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
    min_val = np.min(a)
    return (a + min_val) * mask - min_val

@jit(nopython=True)
def downsample(image: np.ndarray, factor=4):
    """
    Downsamples an image by a given factor
    """
    return image[::factor, ::factor]

@jit(nopython=True)
def gaussian(x_square, sigma):
    return np.exp(-0.5*x_square/sigma**2)

def apply_bilateral_filter(image, sigma_space, sigma_intensity):
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
            result += Gspace * Gintenisity * shifted_image
            # Total weight
            W += Gspace * Gintenisity

    return result / W

def filter_volume(args):
    volume, sigma_space, sigma_intensity = args
    filtered_volume = np.zeros_like(volume)
    for i, slice in enumerate(volume):
        filtered_volume[i] = apply_bilateral_filter(slice, sigma_space, sigma_intensity)
    return filtered_volume

def filter_volume_seq(volume_seq, sigma_space, sigma_intensity):

    total_time = 0
    start_time = time()

    # Create arguments for each volume
    volume_args = [(volume, sigma_space, sigma_intensity) for volume in volume_seq]
    
    # Process volumes in parallel
    with ProcessPoolExecutor() as executor:
        filtered_volumes = list(executor.map(filter_volume, volume_args))
        
    # Update volume sequence with filtered volumes
    volume_seq = np.array(filtered_volumes)
    
    total_time = time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time taken per volume: {total_time / len(volume_seq):.2f} seconds")
    return volume_seq

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
def register_3D_Euler(target_volume: np.ndarray, reference_image: sitk.Image) -> np.ndarray:
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
def register_volume_inplane_weighted(moving_volume: np.ndarray, reference_volume: np.ndarray, n_samples: int = 5, 
               lr: float = 1.0, n_iters: int = 200, relaxation_factor: float = 0.99, 
               gradient_magnitude_tolerance: float = 1e-5, max_step: float = 4.0, min_step: float = 5e-4, 
               weighting_scheme: str = 'inverse', spacing: tuple = (1, 1),
               multi_res: bool = False, smoothing_sigma: float = 2.0,
               verbose: bool = False):
    
    Y, Z, X = moving_volume.shape
    
    if n_samples > Y: n_samples = Y
    elif n_samples < 1: n_samples = 1
    middle = Y // 2
    half_range = n_samples // 2
    slice_indices = np.linspace(middle - half_range, middle + half_range, n_samples, dtype=int)
    
    # Store transforms and their weights
    transforms = []
    metric_values = []

    def command_iteration(method):
        if (method.GetOptimizerIteration() + 1) % 50 == 0:
            print(f"Iteration: {method.GetOptimizerIteration()}")
            print(f"Metric value: {method.GetMetricValue():.4f}")

    # Register each sampled slice
    for slice_idx in slice_indices:
        print(f"Registering slice {slice_idx} of {Y}")
        # Get corresponding slices
        moving_slice = moving_volume[slice_idx]
        reference_slice = reference_volume[slice_idx]
        # Set min pixel value to 0 so that background aligns with pixels moved in by transformation during registration
        min_pixel_value = np.min(moving_slice)
        moving_slice, reference_slice = moving_slice - min_pixel_value, reference_slice - min_pixel_value
        if verbose:
            print(f"{min_pixel_value=}")
        # Convert to SimpleITK images
        moving_image = sitk.GetImageFromArray(moving_slice)
        reference_image = sitk.GetImageFromArray(reference_slice)
        if not multi_res:
            moving_image = sitk.DiscreteGaussian(moving_image, smoothing_sigma)
            reference_image = sitk.DiscreteGaussian(reference_image, smoothing_sigma)
        
        # Set 2D spacing
        moving_image.SetSpacing(spacing)
        reference_image.SetSpacing(spacing)
        
        # Initialize 2D transform
        initial_transform = sitk.CenteredTransformInitializer(
            reference_image,
            moving_image,
            sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        # Setup registration method
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares()
        

        # Optimizer settings
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=lr,
            maximumStepSizeInPhysicalUnits=max_step,
            minStep=min_step,
            numberOfIterations=n_iters,
            gradientMagnitudeTolerance=gradient_magnitude_tolerance,
            relaxationFactor=relaxation_factor,
            
        )
        if multi_res:
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 2])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration_method.SetInitialTransform(initial_transform)
        registration_method.SetInterpolator(sitk.sitkLinear)
        if verbose:
            registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
        try:
            final_transform = registration_method.Execute(reference_image, moving_image)
            metric_value = registration_method.GetMetricValue()
            if verbose:
                print(f"Stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")
                print(f"Metric value: {metric_value:.4f}")
            # Store transform parameters and weight
            params = final_transform.GetParameters()
            center = final_transform.GetCenter()
            transforms.append((params, center))
            metric_values.append(metric_value)

            # Check for extreme transformations
            angle_deg = np.degrees(params[0])  # Convert angle from radians to degrees
            tx, ty = params[1], params[2]
            
            if abs(angle_deg) > 15 or abs(tx) > 10 or abs(ty) > 10:
                print(f"Info: Large transformation detected for slice {slice_idx}:")
                print(f"    Angle: {angle_deg:.2f}°, Translation: ({tx:.2f}, {ty:.2f})")
            
        except RuntimeError as e:
            print(f"Registration failed for slice {slice_idx}: {e}")
            continue
    
    if not transforms:
        print("No successful registrations - returning original volume")
        return moving_volume
    
    # Convert metric values to weights based on chosen scheme
    metric_values = np.array(metric_values)
    print(metric_values)

    if weighting_scheme == 'inverse':
        # Original inverse weighting
        weights = 1.0 / (metric_values + 1e-10)

    elif weighting_scheme == 'inverse_root':
        weights = 1.0 / (np.sqrt(metric_values) + 1e-10)

    elif weighting_scheme == 'softmax':
        # Softmax-based weighting: emphasizes better matches while maintaining non-zero weights
        # Negative because lower metric values are better
        weights = np.exp(-metric_values) / np.sum(np.exp(-metric_values))
    
    elif weighting_scheme == 'threshold':
        # Threshold-based: only keep transforms with metric values below mean
        mean_metric = np.mean(metric_values)
        weights = np.where(metric_values < mean_metric, 1.0, 0.0)
        if np.sum(weights) == 0:  # If all transforms are above mean
            weights = np.ones_like(metric_values)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Compute weighted average transformation
    avg_angle = 0
    avg_tx = 0
    avg_ty = 0
    avg_cx = 0
    avg_cy = 0
    

    for (params, center), weight in zip(transforms, weights):
        angle, tx, ty = params[0], params[1], params[2]
        avg_angle += angle * weight
        avg_tx += tx * weight
        avg_ty += ty * weight
        avg_cx += center[0] * weight
        avg_cy += center[1] * weight
    
    if verbose:
        print(f"{avg_angle=:.4f} {avg_tx=:.4f} {avg_ty=:.4f} {avg_cx=:.4f} {avg_cy=:.4f}")
    # print(f"Time taken: {time() - t1}")
        # Create final average transform
    final_transform = sitk.Euler2DTransform()
    final_transform.SetAngle(avg_angle)
    final_transform.SetTranslation((avg_tx, avg_ty))
    final_transform.SetCenter((avg_cx, avg_cy))
    
    # Apply transform to each slice of moving volume
    registered_volume = np.zeros_like(moving_volume)
    for i in range(moving_volume.shape[0]):
        moving_slice = sitk.GetImageFromArray(moving_volume[i])
        registered_slice = sitk.Resample(
            moving_slice,
            reference_image,
            final_transform,
            sitk.sitkLinear,
            0.0, # min_pixel_value,
            moving_slice.GetPixelID()
        )
        # Bring the slice back to its original value range by adding min_pixel_value
        registered_volume[i] = sitk.GetArrayFromImage(registered_slice) + min_pixel_value

    return registered_volume

def rigid_register_volume_sequence(volume_seq: np.ndarray, 
                                   spacing: tuple = (5.0, 0.488281, 0.488281),
                                   reference_index: int = 1,
                                   n_iters: int = 1000,
                                   n_samples: int = 1,
                                   lr: float = 1.0,
                                   weighting_scheme: str = 'inverse',
                                   verbose: bool = False) -> np.ndarray:
    """
    Performs skull-based rigid registration on a sequence of volumes using the specified reference volume.

    Parameters:
    - volume_seq (np.ndarray): 4D array of shape (T, Y, Z, X) representing the volume sequence.
    - spacing (tuple): Physical spacing of the volume (Y, Z, X). Defaults to (5.0, 0.488281, 0.488281).
    - reference_index (int): Index of the reference volume in the sequence to which other volumes will be registered.
    - registering_structure (str): The structure used in registration. One of: 'all', 'skull', 'brain'. Defaults to 'all'.
    - window_center (int): The center of the window in HU. Defaults to 40.
    - window_width (int): The width of the window in HU. Defaults to 400.

    Returns:
    - np.ndarray: 4D array of registered volumes with the same shape as the input.
    """
    if not (0 <= reference_index < volume_seq.shape[0]):
        raise IndexError(f"Reference index {reference_index} is out of bounds for volume sequence with length {volume_seq.shape[0]}.")
    
    reference_volume = volume_seq[reference_index]

    print(f"Reference volume: {reference_volume.shape}")
    # Initialize the output array
    registered_seq = np.empty_like(volume_seq)
    registered_seq[reference_index] = volume_seq[reference_index]

    # Register each volume sequentially
    for i in range(volume_seq.shape[0]):
        if i == reference_index:
            continue
            
        print(f"Registering volume {i}")
        registered_seq[i] = register_volume_inplane_weighted(
            moving_volume=volume_seq[i],
            reference_volume=reference_volume,
            n_samples=n_samples,
            lr=lr,
            n_iters=n_iters,
            weighting_scheme=weighting_scheme,
            verbose=verbose,
        )

    print("All registrations completed.")
    return registered_seq

def get_volume(folder_path, 
               windowing=True, 
               windowing_type='brain',
               filter=True,
               extract_brain=True,
               standardize=True,
               correct_motion=True,
               reference_index=0,
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
            # if filter: slice = bilateral_filter(slice, 10, 10)
            volume_seq[t, y] = slice

    if extract_brain: # Calculate single brain mask for all volumes before windowing
        volume_mask = get_3d_mask(volume_seq[0])

    if windowing:
        window_center, window_width = get_window_from_type(windowing_type)
        volume_seq = apply_window(volume_seq, window_center, window_width)
        if verbose: ic(volume_seq.max(), volume_seq.min(), volume_seq.dtype)

    if filter: volume_seq = filter_volume_seq(volume_seq, 20, 10)
    
    if correct_motion:
        volume_seq = rigid_register_volume_sequence(
            volume_seq, 
            spacing=spacing, 
            reference_index=reference_index,
            n_iters=1000,
            n_samples=1,
        )

    if extract_brain: # Apply brain mask to all registered volumes
        volume_seq = apply_mask(volume_seq, volume_mask)

    if verbose: print(f"Done!")
    # Standardization
    if standardize:
        return (volume_seq - np.mean(volume_seq)) / np.std(volume_seq)
    return volume_seq

def save_volume(volume, folder_path='volume.npy'):
    np.save(folder_path, volume)
    ic(f"Volume saved to {folder_path}")

def save_all_volumes(num_samples=10, scan_size='small'):
    # Take all volumes with scan size 'small' --> 18x16x512x512 (T, Y, Z, X)
    for folder_path in load_folder_paths(scan_size=scan_size)[:num_samples]:
        volume_seq = get_volume(folder_path)
        save_volume(volume_seq, folder_path.replace('.dcm', '.npy'))

def save_selected_volumes(IDs: list, save_path: str='selected_volumes'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ID in IDs:
        if os.path.exists(os.path.join(save_path, f'MOL-{ID}.npy')):
            print(f"Volume {ID} already exists")
            continue
        folder_path = os.path.join(dataset_path, f'MOL-{ID}')
        volume_seq = get_volume(folder_path, spatial_downsampling_factor=2)
        save_volume(volume_seq, os.path.join(save_path, f'MOL-{ID}.npy'))

if __name__ == "__main__":
    IDs = ['001', '060', '061', '062', '094','096', '097', '101', '104', '105']
    save_selected_volumes(IDs)