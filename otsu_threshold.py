import numpy as np
import matplotlib.pyplot as plt
import cv2

class OtsuThreshold:
    """
    Class implementing Otsu thresholding for image segmentation.
    Provides methods for both global and local thresholding approaches.
    """
    
    @staticmethod
    def calculate_otsu_threshold(histogram, total_pixels):
        """
        Calculate the optimal threshold using Otsu's method
        
        Args:
            histogram: Array of histogram values (256 bins)
            total_pixels: Total number of pixels in the image/region
            
        Returns:
            optimal_threshold: The calculated optimal threshold
            max_variance: The maximum between-class variance
        """
        # Initialize variables
        sum_total = 0
        for i in range(256):
            sum_total += i * histogram[i]
            
        # Variables for the calculation
        weight_background = 0
        sum_background = 0
        max_variance = 0
        optimal_threshold = 0
        
        # Iterate through all possible thresholds
        for threshold in range(256):
            # Update weights and means for background and foreground
            weight_background += histogram[threshold]
            if weight_background == 0:
                continue
                
            weight_foreground = total_pixels - weight_background
            if weight_foreground == 0:
                break
                
            sum_background += threshold * histogram[threshold]
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
            
            # Calculate between-class variance
            between_variance = weight_background * weight_foreground * \
                              (mean_background - mean_foreground) ** 2
                              
            # Update optimal threshold if current variance is higher
            if between_variance > max_variance:
                max_variance = between_variance
                optimal_threshold = threshold
                
        return optimal_threshold, max_variance
    
    @staticmethod
    def global_otsu(image):
        """
        Apply global Otsu thresholding to the entire image
        
        Args:
            image: Grayscale input image
            
        Returns:
            thresholded_image: Binary output image
        """
        # Make a copy of the image and ensure it's grayscale
        src = np.copy(image)
        if len(src.shape) > 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            
        # Calculate histogram
        hist = np.zeros(256)
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                hist[src[i, j]] += 1
                
        # Total number of pixels
        total_pixels = src.shape[0] * src.shape[1]
        
        # Calculate Otsu threshold
        optimal_threshold, _ = OtsuThreshold.calculate_otsu_threshold(hist, total_pixels)
        
        # Apply threshold to create binary image
        thresholded_image = np.zeros_like(src)
        thresholded_image[src > optimal_threshold] = 255
        
        return thresholded_image
    
    @staticmethod
    def local_otsu(image, window_size):
        """
        Apply local Otsu thresholding using a sliding window approach
        
        Args:
            image: Grayscale input image
            window_size: Size of the local window for processing
            
        Returns:
            thresholded_image: Binary output image
        """
        # Make a copy of the image and ensure it's grayscale
        src = np.copy(image)
        if len(src.shape) > 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            
        # Create output image
        thresholded_image = np.zeros_like(src)
        
        # Get image dimensions
        rows, cols = src.shape
        
        # Calculate half window size for padding
        half_window = window_size // 2
        
        # Process the image with a sliding window
        for i in range(rows):
            for j in range(cols):
                # Define window boundaries with boundary checks
                row_start = max(0, i - half_window)
                row_end = min(rows, i + half_window + 1)
                col_start = max(0, j - half_window)
                col_end = min(cols, j + half_window + 1)
                
                # Extract window
                window = src[row_start:row_end, col_start:col_end]
                
                # Calculate histogram for the window
                hist = np.zeros(256)
                for r in range(window.shape[0]):
                    for c in range(window.shape[1]):
                        hist[window[r, c]] += 1
                
                # Total pixels in window
                window_pixels = window.shape[0] * window.shape[1]
                
                # Calculate Otsu threshold for the window
                local_threshold, _ = OtsuThreshold.calculate_otsu_threshold(hist, window_pixels)
                
                # Apply threshold to the current pixel
                if src[i, j] > local_threshold:
                    thresholded_image[i, j] = 255
        
        return thresholded_image
    
    @staticmethod
    def optimized_local_otsu(image, window_size):
        """
        Optimized version of local Otsu thresholding using a block-processing approach
        
        Args:
            image: Grayscale input image
            window_size: Size of the local windows for processing
            
        Returns:
            thresholded_image: Binary output image
        """
        # Make a copy of the image and ensure it's grayscale
        src = np.copy(image)
        if len(src.shape) > 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            
        # Create output image
        thresholded_image = np.zeros_like(src)
        
        # Get image dimensions
        rows, cols = src.shape
        
        # Process the image in blocks
        for i in range(0, rows, window_size):
            for j in range(0, cols, window_size):
                # Define block boundaries with boundary checks
                block_end_i = min(i + window_size, rows)
                block_end_j = min(j + window_size, cols)
                
                # Extract block
                block = src[i:block_end_i, j:block_end_j]
                
                # Calculate histogram for the block
                hist = np.zeros(256)
                for r in range(block.shape[0]):
                    for c in range(block.shape[1]):
                        hist[block[r, c]] += 1
                
                # Total pixels in block
                block_pixels = block.shape[0] * block.shape[1]
                
                # Calculate Otsu threshold for the block
                if block_pixels > 0:  # Ensure we have pixels to process
                    block_threshold, _ = OtsuThreshold.calculate_otsu_threshold(hist, block_pixels)
                    
                    # Apply threshold to the current block
                    thresholded_image[i:block_end_i, j:block_end_j] = np.where(
                        block > block_threshold, 255, 0
                    )
        
        return thresholded_image
    
    @staticmethod
    def applySegmentation(image, mode, window_size=1):
        """
        Apply Otsu thresholding based on the specified mode
        
        Args:
            image: Grayscale input image
            mode: Thresholding mode ('Global' or 'Local')
            window_size: Size of the local window for processing (used only in Local mode)
            
        Returns:
            thresholded_image: Binary output image
        """
        if mode == "Global":
            return OtsuThreshold.global_otsu(image)
        elif mode == "Local":
            # Use the optimized version for better performance
            return OtsuThreshold.optimized_local_otsu(image, window_size)
        else:
            raise ValueError(f"Unknown thresholding mode: '{mode}'. Use 'Global' or 'Local'.")

