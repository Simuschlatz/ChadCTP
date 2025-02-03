if __name__ == "__main__":
    from preprocessing import get_volume
    from scipy import ndimage
    import os
    import SimpleITK as sitk
    import numpy as np
    from multiprocessing import freeze_support

    dataset_path = os.path.expanduser('~/Desktop/UniToBrain')

    from skimage import morphology
    from scipy import ndimage
    from numba import jit

    @jit(nopython=True)
    def get_threshold_mask(slice, min_val=-40, max_val=150):
        return (min_val < slice) & (slice < max_val)

    @jit(nopython=True)
    def apply_mask(a: np.ndarray, mask: np.ndarray):
        min_val = np.min(a)
        return (a - min_val) * mask + min_val

    def mask_slice(slice, threshold_min=-40, threshold_max=150, structuring_element_dims=(3, 3)):
        smoothened = ndimage.gaussian_filter(slice, sigma=2.0)
        mask = get_threshold_mask(smoothened, min_val=threshold_min, max_val=threshold_max)
        mask = ndimage.binary_erosion(mask, np.ones(structuring_element_dims), iterations=2)
        # mask = ndimage.binary_erosion(mask, np.ones((3, 3)))
        # Remove small objects
        mask = morphology.remove_small_objects(mask, 800)
        # Fill small holes
        mask = ndimage.binary_dilation(mask, np.ones(structuring_element_dims), iterations=2)
        mask = ndimage.binary_fill_holes(mask)
        return apply_mask(slice, mask)

    def preprocess_slice_for_registration(slice, threshold_min=-40, threshold_max=150, structuring_element_dims=(3, 3), reg_window_min=20, reg_window_max=60):
        masked = mask_slice(slice, threshold_min=threshold_min, threshold_max=threshold_max, structuring_element_dims=structuring_element_dims)
        return np.clip(masked, reg_window_min, reg_window_max)

    def experiment_reg(moving_volume: np.ndarray, reference_volume: np.ndarray, n_samples: int = 5, 
                lr: float = 1.0, n_iters: int = 200, relaxation_factor: float = 0.99, multi_res: bool = False,
                gradient_magnitude_tolerance: float = 1e-5, max_step: float = 4.0, min_step: float = 5e-4, 
                spacing: tuple = (1, 1), smoothing_sigma: float = 2.0, interpolator=sitk.sitkLinear,
                mask=True, reg_window_min=-100, reg_window_max=300, threshold_min=0, threshold_max=500,
                verbose: bool = False):
        
        Y, Z, X = moving_volume.shape
        
        if n_samples > Y: n_samples = Y
        elif n_samples < 1: n_samples = 1
        middle = Y // 2
        half_range = n_samples // 2
        slice_indices = np.linspace(middle - half_range, middle + half_range, n_samples, dtype=int)
        
        mean_metric_value = 0
        transforms = []
        weights = np.zeros(n_samples)
        
        # Register each sampled slice
        for pos, slice_idx in enumerate(slice_indices):
            
            # Standard preprocessing for registration
            if mask:
                moving_slice = preprocess_slice_for_registration(moving_volume[slice_idx], reg_window_min=reg_window_min, reg_window_max=reg_window_max, threshold_min=threshold_min, threshold_max=threshold_max)
                fixed_slice = preprocess_slice_for_registration(reference_volume[slice_idx], reg_window_min=reg_window_min, reg_window_max=reg_window_max, threshold_min=threshold_min, threshold_max=threshold_max)
            else:
                moving_slice = np.clip(moving_volume[slice_idx], 0, 160)
                fixed_slice = np.clip(reference_volume[slice_idx], 0, 160)
                

            # Set min pixel value to 0 so that background aligns with pixels moved in by transformation during registration
            min_pixel_value = np.min(moving_slice)
            moving_slice, fixed_slice = moving_slice - min_pixel_value, fixed_slice - min_pixel_value

            # Convert to SimpleITK images
            moving_image = sitk.GetImageFromArray(moving_slice)
            fixed_image = sitk.GetImageFromArray(fixed_slice)
            
            if smoothing_sigma:
                moving_image = sitk.DiscreteGaussian(moving_image, smoothing_sigma)
                fixed_image = sitk.DiscreteGaussian(fixed_image, smoothing_sigma)
            
            # Set 2D spacing
            moving_image.SetSpacing(spacing)
            fixed_image.SetSpacing(spacing)
            
            # Initialize 2D transform
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image,
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
                registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[2, 4])
                registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 2])
                registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration_method.SetInitialTransform(initial_transform)
            registration_method.SetInterpolator(interpolator)
    
            try:
                final_transform = registration_method.Execute(fixed_image, moving_image)
                params = final_transform.GetParameters()
                transforms.append(params)

                metric_value = registration_method.GetMetricValue()
                mean_metric_value += metric_value / n_samples
                position_weight = 1 - ((slice_idx - middle) / (middle-1)) # The further away from the center, the lower the weight
                weight = 1 / ((metric_value) + 1e-10) * position_weight ** 1.5
                weights[pos] = weight

            except RuntimeError as e:
                print(f"Registration failed for slice {slice_idx}: {e}")
                continue
        
        # Normalize weights
        weights = weights / np.sum(weights)
        # print(weights)
        # Compute weighted average transformation
        avg_angle = 0
        avg_tx = 0
        avg_ty = 0

        for params, weight in zip(transforms, weights):
            angle, tx, ty = params[0], params[1], params[2]
            avg_angle += angle * weight
            avg_tx += tx * weight
            avg_ty += ty * weight


            # Create final average transform
        final_transform = sitk.Euler2DTransform()
        final_transform.SetAngle(avg_angle)
        final_transform.SetTranslation((avg_tx, avg_ty))
        final_transform.SetCenter((127, 127))
        
        # Apply transform to each slice of moving volume
        registered_volume = np.zeros_like(moving_volume)
        for i in range(moving_volume.shape[0]):
            moving_slice = moving_volume[i]
            min_value = np.min(moving_slice)
            moving_slice = sitk.GetImageFromArray(moving_slice - min_value)
            registered_slice = sitk.Resample(
                moving_slice,
                final_transform,
                interpolator,
                0.0, # Min pixel value
                moving_slice.GetPixelID()
            )
            # Bring the slice back to its original value range by adding min_pixel_value
            registered_volume[i] = sitk.GetArrayFromImage(registered_slice) + min_value
        return registered_volume

    def mse(moving_volume: np.ndarray, reference_volume: np.ndarray):
        return np.mean(np.square(moving_volume - reference_volume))

    import json
    def write_results(results: dict):
        with open("registration_results.json", "w") as f:
            json.dump(results, f)


    learning_rates = [2.0, 1.5, 1.0, 0.5]
    n_samples = [1, 3, 5, 7, 9]
    relaxation_factors = [0.99, 0.95, 0.9]
    smoothing_sigmas = [2.0, 1.5, 1.0, 0.5, 0]
    iterations = [100, 200, 500, 1000]
    window_min_max = [(0, 80), (0, 160), (20, 60), (-100, 300)]
    interpolators = [sitk.sitkLinear, sitk.sitkBSpline, sitk.sitkNearestNeighbor]
    ref_index = [1, 2, 3]
    mask = [True, False]
    scan_ids = ["/MOL-" + s for s in ['062', '063', '092', '098', '104', '133']]
    paths = [dataset_path + sid for sid in scan_ids]
    print("loading scans")
    scans = []
    for path in paths:
        scans.append(get_volume(path, 
                    extract_brain=False, 
                    filter=True, 
                    window_params=None, 
                    correct_motion=False, 
                    standardize=False, 
                    spatial_downsampling_factor=2, 
                    temporal_downsampling_factor=1,
                    verbose=False
                    )
                    )
    print("running experiments")
    results = {}
    # Experiment
    for lr in learning_rates:
        for i in iterations:
            for n in n_samples:
                for relax in relaxation_factors:
                    for multi_res in [True, False]:
                        for sigma in smoothing_sigmas if not multi_res else [0]:
                            for interp in interpolators:
                                for (window_min, window_max) in window_min_max:
                                    for m in mask:
                                        for v_seq in scans:
                                            for ref_i in ref_index:
                                                results[(ref_i, lr, n, relax, sigma, interp, m, (window_min, window_max))] = []
                                                v_ref = v_seq[ref_i]
                                                mean_metric = 0
                                                for moving_idx in range(len(v_seq)):
                                                    if moving_idx == ref_i:
                                                        continue
                                                    registered = experiment_reg(
                                                                                v_seq[moving_idx], 
                                                                                v_ref, 
                                                                                lr=lr, 
                                                                                n_iters=i,
                                                                                n_samples=n, 
                                                                                relaxation_factor=relax,
                                                                                multi_res=multi_res,
                                                                                smoothing_sigma=sigma, 
                                                                                interpolator=interp,
                                                                                mask=m,
                                                                                reg_window_min=window_min,
                                                                                reg_window_max=window_max
                                                                                )
                                                    metric = mse(np.clip(registered, -10, 60), np.clip(v_ref, -10, 60))
                                                    mean_metric += metric / (len(v_seq) - 1)
                                                results[(ref_i, lr, n, relax, sigma, interp, m, (window_min, window_max))].append(mean_metric)
            write_results(results)