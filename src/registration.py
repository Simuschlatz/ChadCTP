import numpy as np
import SimpleITK as sitk
import time
from scipy import ndimage

def register_volume_inplane_weighted(moving_volume: np.ndarray, reference_volume: np.ndarray, 
                      n_samples: int = 5, lr: float = 5, n_iters: int = 1000, 
                      spacing: tuple = (5.0, 0.488281, 0.488281), 
                      weighting_scheme: str = 'inverse') -> np.ndarray:
    """
    Performs in-plane registration by:
    1. Sampling n evenly-spaced slices
    2. Performing 2D registration on each slice
    3. Averaging transformations weighted by their metric values
    4. Applying final transformation to entire volume
    
    Args:
        moving_volume: Volume to be registered (Y, Z, X)
        reference_volume: Reference volume (Y, Z, X)
        n_samples: Number of slices to sample for registration
        lr: Learning rate for registration
        n_iters: Number of iterations for registration
        spacing: Physical spacing of the volume (Y, Z, X)
    """
    Y, Z, X = moving_volume.shape
    
    # Sample slice indices evenly
    slice_indices = np.linspace(Y//4, 3*Y//4, n_samples, dtype=int)
    
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
        
        # Convert to SimpleITK images
        moving_image = sitk.GetImageFromArray(moving_slice)
        reference_image = sitk.GetImageFromArray(reference_slice)
        
        # Set 2D spacing
        moving_image.SetSpacing(spacing[1:])
        reference_image.SetSpacing(spacing[1:])
        
        # Create smoothed versions for registration
        smoothed_moving = sitk.DiscreteGaussian(moving_image, 1.0)
        smoothed_reference = sitk.DiscreteGaussian(reference_image, 1.0)
        
        # Initialize 2D transform
        initial_transform = sitk.CenteredTransformInitializer(
            smoothed_reference,
            smoothed_moving,
            sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
        
        # Setup registration method
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=lr,
            minStep=1e-4,
            numberOfIterations=n_iters,
            gradientMagnitudeTolerance=1e-8,
        )
        registration_method.SetInitialTransform(initial_transform)
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

        try:
            final_transform = registration_method.Execute(smoothed_reference, smoothed_moving)
            metric_value = registration_method.GetMetricValue()
            
            # Store transform parameters and weight
            params = final_transform.GetParameters()
            center = final_transform.GetCenter()
            transforms.append((params, center))
            metric_values.append(metric_value)
            
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
    
    elif weighting_scheme == 'exponential':
        # Exponential decay: w = exp(-metric_value)
        # More robust to outliers than inverse
        weights = np.exp(-metric_values)
    
    elif weighting_scheme == 'softmax':
        # Softmax-based weighting: emphasizes better matches while maintaining non-zero weights
        # Negative because lower metric values are better
        weights = np.exp(-metric_values) / np.sum(np.exp(-metric_values))
        
    elif weighting_scheme == 'rank':
        # Rank-based weighting: less sensitive to absolute metric values
        ranks = np.argsort(np.argsort(-metric_values))  # Higher rank for lower metric value
        weights = 1.0 / (ranks + 1)
    
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
        avg_angle += params[0] * weight
        avg_tx += params[1] * weight
        avg_ty += params[2] * weight
        avg_cx += center[0] * weight
        avg_cy += center[1] * weight
    
    print(avg_angle, avg_tx, avg_ty, avg_cx, avg_cy)


    # Precompute trig values
    cos_theta = np.cos(avg_angle)
    sin_theta = np.sin(avg_angle)

    # Build affine matrix once
    affine_matrix = np.array([
        [cos_theta, -sin_theta, avg_tx + avg_cx - avg_cx*cos_theta + avg_cy*sin_theta],
        [sin_theta, cos_theta, avg_ty + avg_cy - avg_cx*sin_theta - avg_cy*cos_theta],
        [0, 0, 1]
    ])

    # Extract transformation components
    transform_matrix = affine_matrix[:2, :2]
    offset = affine_matrix[:2, 2]
    # Apply transform to entire volume slice by slice
    registered_volume = np.zeros_like(moving_volume)
    t1 = time()
    for y in range(Y):
        registered_volume[y] = ndimage.affine_transform(
            moving_volume[y],
            transform_matrix,
            offset=offset,
            output_shape=moving_volume[y].shape,
            order=1
        )
    print(f"Time taken: {time() - t1}")

    return registered_volume