import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

class SpectralThreshold:

    @staticmethod
    def apply_spectral_threshold(input_image : np.ndarray):
        # Make a copy of the input source image
        src = np.copy(input_image)
        if len(src.shape) > 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        else:
            pass

        # Get input image dimensions
        YDim, XDim = src.shape
        # Get the histogram bins values
        HistValues = plt.hist(src.ravel(), 256)[0]
        # Calculate PDF
        PDF = HistValues / (YDim * XDim)
        # Calculate CDF
        CDF = np.cumsum(PDF)
        optimal_low = 1
        optimal_high = 1
        max_variance = 0
        # Loop over all the possible thresholds and select the one with maximum variance between the background and the object
        Global = np.arange(0, 256)
        global_mean = sum(Global * PDF) / CDF[-1]
        for low_threshold in range(1, 254):
            for hight_threshold in range(low_threshold + 1, 255):
                try:
                    # array for background intensities
                    back_intensity_array = np.arange(0, low_threshold)
                    # array for low intensities
                    low_intensity_array = np.arange(low_threshold, hight_threshold)
                    # array for high intensities
                    high_intensity_array = np.arange(hight_threshold, 256)
                    # Get low intensities CDF
                    CDF_low_intenisty = np.sum(PDF[low_threshold:hight_threshold])
                    # Get low intensities CDF
                    CDF_high_intenisty = np.sum(PDF[hight_threshold:256])
                    # Calculation mean of background & the object, based on CDF & PDF
                    back_intensity_mean = sum(back_intensity_array * PDF[0:low_threshold]) / CDF[low_threshold]
                    if CDF_low_intenisty != 0:
                        low_intensity_mean = sum(low_intensity_array * PDF[low_threshold:hight_threshold]) / CDF_low_intenisty
                    else:
                        low_intensity_mean = 0  
                    high_intensity_mean = sum(high_intensity_array * PDF[hight_threshold:256]) / CDF_high_intenisty
                    # Calculate cross-class variance
                    variance = (CDF[low_threshold] * (back_intensity_mean - global_mean) ** 2 + (CDF_low_intenisty * (low_intensity_mean - global_mean) ** 2) + (
                            CDF_high_intenisty * (high_intensity_mean - global_mean) ** 2))
                    # Filter the max variance & it's Threshold
                    if variance > max_variance:
                        max_variance = variance
                        optimal_low = low_threshold
                        optimal_high = hight_threshold
                except RuntimeWarning:
                    pass

        # Create empty array
        thresholded_image = np.zeros(src.shape)

        strong = 255
        weak = 128

        # Find position of strong & weak pixels
        StrongRow, StrongCol = np.where(src >= optimal_high)
        WeakRow, WeakCol = np.where((src <= optimal_high) & (src >= optimal_low))

        # Apply thresholding
        thresholded_image[StrongRow, StrongCol] = strong
        thresholded_image[WeakRow, WeakCol] = weak

        return thresholded_image
    

    @staticmethod
    def LocalThresholding(input_image : np.ndarray, window_size):
        # Make a copy of the input source image
        src = np.copy(input_image)

        # If the image has more than one channel, convert it to grayscale
        if len(src.shape) > 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        else:
            pass  

        # Get the dimensions of the source image
        YDim, XDim = src.shape

        # Initialize an empty thresholded_image image of the same size
        thresholded_part = np.zeros((YDim, XDim))

        # Calculate the step size for each window along X and Y axes
        YStep = YDim // window_size
        XStep = XDim // window_size

        # Create lists to store the range boundaries for each window
        XRange = []
        YRange = []

        # Populate XRange and YRange with starting points for each window
        for i in range(0, window_size):
            XRange.append(XStep * i)
        for i in range(0, window_size):
            YRange.append(YStep * i)

        # Append the maximum boundary to cover the full image
        XRange.append(XDim)
        YRange.append(YDim)

        # Apply local spectral thresholding for each window block
        for x in range(0, window_size):
            for y in range(0, window_size):
                # Apply thresholding to the current window and save it into the thresholded_part image
                thresholded_part[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]] = SpectralThreshold.apply_spectral_threshold(
                    src[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]]
                )

        # Return the thresholded result
        return thresholded_part


    @staticmethod
    def applySegmentation(img, mode, window_size):
        # Function to choose the mode 
        if mode == "Global":
            return SpectralThreshold.apply_spectral_threshold(img)
        elif mode == "Local":
            return SpectralThreshold.LocalThresholding(img, window_size)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'global' or 'local'.")
