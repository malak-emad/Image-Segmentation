import numpy as np

def apply_mean_shift_fast(image, window_size, convergence_threshold, sample_fraction=0.01):

    # Step 1: Calc no. pixels
    height, width = image.shape[:2]
    pixels = image.reshape((-1, 3)).astype(np.float32)
    num_pixels = pixels.shape[0]
    
    # Step 2: Randomly sample a fraction of pixels
    sample_size = int(sample_fraction * num_pixels)
    sample_indices = np.random.choice(num_pixels, sample_size, replace=False)
    sampled_pixels = pixels[sample_indices]
    
    #Apply mean-shift only on sampled pixels
    #Step 3: initalize modes by zeros
    modes = np.zeros_like(sampled_pixels, dtype=np.float32)
    
    #for each sampled pixel, initialize current mean by its RGB, prev mean by zero and iterations by zero
    for i in range(len(sampled_pixels)):
        current_mean = sampled_pixels[i].copy()
        prev_mean = np.zeros_like(current_mean)
        iterations = 0
        
        #Step 4:as long as difference between current and previous means > Threshold
        while np.sqrt(np.sum((current_mean - prev_mean)**2)) > convergence_threshold:
            prev_mean = current_mean.copy() #update prev mean
            #Step 5: Calc distances between sampled pixels and current mean 
            distances = np.sqrt(np.sum((sampled_pixels - current_mean)**2, axis=1)) 
            #Step 6: check for pixels within window size and calc mean for them
            within_window = distances < window_size
            if np.sum(within_window) > 0:
                current_mean = np.mean(sampled_pixels[within_window], axis=0)
            iterations += 1
            if iterations > 100:
                break
        modes[i] = current_mean

        if i % 100 == 0:
            print(f"Processed {i}/{len(sampled_pixels)} sampled pixels")
    
    # Step 7: Cluster sampled pixels
    modes_rounded = np.round(modes, decimals=1)
    unique_modes, inverse_indices = np.unique(modes_rounded, axis=0, return_inverse=True)
    
    # Step 8: Assign every pixel in original image to nearest cluster
    result = np.zeros_like(pixels)
    for i in range(num_pixels):
        distances = np.sqrt(np.sum((unique_modes - pixels[i])**2, axis=1))
        nearest_mode_idx = np.argmin(distances)
        result[i] = unique_modes[nearest_mode_idx]
    
    result_image = result.reshape((height, width, 3)).astype(np.uint8)
    print(f"Mean-shift (fast) found {len(unique_modes)} clusters")
    return result_image


def apply_meanshift(image, window_size, convergence_threshold):

    # Convert parameters if they're strings
    if isinstance(window_size, str):
        window_size = float(window_size)
    if isinstance(convergence_threshold, str):
        convergence_threshold = float(convergence_threshold)
    
    print("Using standard mean-shift")
    return apply_mean_shift_fast(image, window_size, convergence_threshold)