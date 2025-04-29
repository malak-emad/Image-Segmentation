import cv2
import numpy as np

class OptimalThreshold:
    """
    Implementation of Optimal Thresholding technique for image segmentation.
    Can operate in both Global and Local modes.
    """
    
    @staticmethod
    def apply_global_threshold(image):
        """
        Applies global optimal thresholding to a grayscale image
        
        Args:
            image: NumPy array of grayscale image
            
        Returns:
            Binary thresholded image
        """
        # Make a copy to avoid modifying original
        src = np.copy(image)
        
        # Convert to grayscale if needed
        if len(src.shape) > 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        
        # Calculate initial threshold (averaging 4 corners as background and rest as foreground)
        old_threshold = OptimalThreshold._get_initial_threshold(src)
        
        # Iteratively refine the threshold
        new_threshold = OptimalThreshold._get_optimal_threshold(src, old_threshold)
        iteration = 0
        max_iterations = 100  # Safety to prevent infinite loops
        
        # Iterate until threshold stabilizes or max iterations reached
        while old_threshold != new_threshold and iteration < max_iterations:
            old_threshold = new_threshold
            new_threshold = OptimalThreshold._get_optimal_threshold(src, old_threshold)
            iteration += 1
            
        # Apply global threshold using the computed optimal value
        result = np.zeros_like(src, dtype=np.uint8)
        result[src >= new_threshold] = 255
        
        return result
    
    @staticmethod
    def apply_local_threshold(image, window_size):
        """
        Applies local optimal thresholding to a grayscale image by dividing
        the image into regions of window_size x window_size
        
        Args:
            image: NumPy array of grayscale image
            window_size: Size of local regions 
            
        Returns:
            Binary thresholded image
        """
        # Make a copy to avoid modifying original
        src = np.copy(image)
        
        # Convert to grayscale if needed
        if len(src.shape) > 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        
        # Get image dimensions
        height, width = src.shape
        
        # Calculate the number of regions in x and y directions
        regions_y = height // window_size
        regions_x = width // window_size
        
        # Add extra region if there's a remainder
        if height % window_size != 0:
            regions_y += 1
        if width % window_size != 0:
            regions_x += 1
        
        # Create output image
        result = np.zeros_like(src, dtype=np.uint8)
        
        # Process each local region
        for y in range(regions_y):
            y_start = y * window_size
            y_end = min((y + 1) * window_size, height)
            
            for x in range(regions_x):
                x_start = x * window_size
                x_end = min((x + 1) * window_size, width)
                
                # Extract local region
                local_region = src[y_start:y_end, x_start:x_end]
                
                # Skip very small regions or uniform regions
                if local_region.size <= 1 or np.min(local_region) == np.max(local_region):
                    continue
                
                # Apply optimal thresholding to local region
                try:
                    # Calculate initial threshold
                    init_threshold = OptimalThreshold._get_initial_threshold(local_region)
                    
                    # Iteratively refine the threshold
                    old_threshold = init_threshold
                    new_threshold = OptimalThreshold._get_optimal_threshold(local_region, old_threshold)
                    iteration = 0
                    max_iterations = 50  # Safety to prevent infinite loops
                    
                    # Iterate until threshold stabilizes
                    while old_threshold != new_threshold and iteration < max_iterations:
                        old_threshold = new_threshold
                        new_threshold = OptimalThreshold._get_optimal_threshold(local_region, old_threshold)
                        iteration += 1
                    
                    # Apply threshold to local region
                    result[y_start:y_end, x_start:x_end] = np.where(
                        local_region >= new_threshold, 255, 0
                    )
                except:
                    # In case of any error (e.g., division by zero), skip this region
                    continue
        
        return result
    
    @staticmethod
    def _get_initial_threshold(image):
        """
        Gets the initial threshold by assuming the four corners are background
        and the rest is foreground, as described in the algorithm.
        """
        # Get image dimensions
        height, width = image.shape
        
        # Calculate background mean from the four corners
        background_mean = (
            int(image[0, 0]) + 
            int(image[0, width-1]) + 
            int(image[height-1, 0]) + 
            int(image[height-1, width-1])
        ) / 4
        
        # Calculate foreground mean from all other pixels
        total_sum = np.sum(image)
        corner_sum = int(image[0, 0]) + int(image[0, width-1]) + int(image[height-1, 0]) + int(image[height-1, width-1])
        foreground_sum = total_sum - corner_sum
        foreground_pixels = image.size - 4
        foreground_mean = foreground_sum / foreground_pixels if foreground_pixels > 0 else 0
        
        # Initial threshold is average of background and foreground means
        threshold = (background_mean + foreground_mean) / 2
        
        return threshold
    
    @staticmethod
    def _get_optimal_threshold(image, threshold):
        """
        Calculates optimal threshold based on the given initial threshold.
        Implements the algorithm as shown in the image.
        """
        # Separate pixels into background and foreground based on current threshold
        background = image[image < threshold]
        foreground = image[image >= threshold]
        
        # Handle empty classes
        if background.size == 0:
            background_mean = 0
        else:
            background_mean = np.mean(background)
            
        if foreground.size == 0:
            foreground_mean = 255
        else:
            foreground_mean = np.mean(foreground)
        
        # Calculate new threshold as average of class means
        optimal_threshold = (background_mean + foreground_mean) / 2
        
        return optimal_threshold
    
    @staticmethod
    def applySegmentation(image, mode="Global", window_size=8):
        """
        Main method to apply optimal thresholding segmentation
        
        Args:
            image: Input image (grayscale or RGB)
            mode: "Global" or "Local" thresholding mode
            window_size: Size of local windows (only used if mode is "Local")
            
        Returns:
            Binary thresholded image
        """
        if mode == "Global":
            return OptimalThreshold.apply_global_threshold(image)
        elif mode == "Local":
            return OptimalThreshold.apply_local_threshold(image, window_size)
        else:
            raise ValueError(f"Unknown thresholding mode: {mode}")