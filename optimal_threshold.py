import numpy as np
import cv2

class OptimalThreshold:
    @staticmethod
    def apply_global_thresholding(image):
        """
        Apply global optimal thresholding to a grayscale image.
        
        Args:
            image: Grayscale input image as numpy array
            
        Returns:
            Binary segmented image
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Step 1: Initialize threshold using the four corners as background
        h, w = image.shape
        corners = [image[0, 0], image[0, w-1], image[h-1, 0], image[h-1, w-1]]
        t_old = np.mean(corners)
        
        # Convergence criteria
        max_iterations = 100
        epsilon = 0.5
        
        for _ in range(max_iterations):
            # Step 2: Segment image using current threshold
            background = image[image <= t_old]
            objects = image[image > t_old]
            
            # Check if we have pixels in both classes
            if len(background) == 0 or len(objects) == 0:
                break
                
            # Calculate mean of each class
            mu_b = np.mean(background)
            mu_o = np.mean(objects)
            
            # Step 3: Update threshold
            t_new = (mu_b + mu_o) / 2
            
            # Step 4: Check convergence
            if abs(t_new - t_old) < epsilon:
                break
                
            t_old = t_new
        
        # Apply final threshold
        binary_image = np.zeros_like(image)
        binary_image[image > t_old] = 255
        
        return binary_image
    
    @staticmethod
    def apply_local_thresholding(image, window_size):
        """
        Apply local optimal thresholding to a grayscale image.
        
        Args:
            image: Grayscale input image as numpy array
            window_size: Size of the local window
            
        Returns:
            Binary segmented image
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        h, w = image.shape
        binary_image = np.zeros_like(image)
        
        # Validate window size
        if window_size % 2 == 0:
            window_size += 1  # Ensure window size is odd
        
        half_window = window_size // 2
        
        # Process each pixel with local window
        for i in range(h):
            for j in range(w):
                # Define local window boundaries
                row_start = max(0, i - half_window)
                row_end = min(h, i + half_window + 1)
                col_start = max(0, j - half_window)
                col_end = min(w, j + half_window + 1)
                
                # Extract local window
                local_window = image[row_start:row_end, col_start:col_end]
                
                # Apply optimal thresholding to local window
                local_mean = np.mean(local_window)
                
                # Simplified version for local windows:
                # Instead of iterative process, we'll use a simpler approach for local windows
                # Take pixels below and above the mean for initial classes
                background = local_window[local_window <= local_mean]
                objects = local_window[local_window > local_mean]
                
                # Check if we have pixels in both classes
                if len(background) == 0 or len(objects) == 0:
                    threshold = local_mean
                else:
                    # Calculate mean of each class
                    mu_b = np.mean(background)
                    mu_o = np.mean(objects)
                    
                    # Update threshold
                    threshold = (mu_b + mu_o) / 2
                
                # Apply threshold to current pixel
                if image[i, j] > threshold:
                    binary_image[i, j] = 255
        
        return binary_image

    @staticmethod
    def applySegmentation(image, mode="Global", window_size=15):
        """
        Main interface method to apply optimal thresholding segmentation
        
        Args:
            image: Input image (grayscale or RGB)
            mode: "Global" or "Local" thresholding
            window_size: Size of local window (only used for local thresholding)
            
        Returns:
            Binary segmented image
        """
        if mode == "Global":
            return OptimalThreshold.apply_global_thresholding(image)
        elif mode == "Local":
            return OptimalThreshold.apply_local_thresholding(image, window_size)
        else:
            raise ValueError(f"Unknown thresholding mode: {mode}")