import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

class SpectralThreshold:

    @staticmethod
    def apply_spectral_threshold(source: np.ndarray):
        """
        Applies Thresholding To The Given Grayscale Image Using Spectral Thresholding Method
        :param source: NumPy Array of The Source Grayscale Image
        :return: Thresholded Image
        """
        src = np.copy(source)
        if len(src.shape) > 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        else:
            pass

        # Get Image Dimensions
        YRange, XRange = src.shape
        # Get The Values of The Histogram Bins
        HistValues = plt.hist(src.ravel(), 256)[0]
        # Calculate The Probability Density Function
        PDF = HistValues / (YRange * XRange)
        # Calculate The Cumulative Density Function
        CDF = np.cumsum(PDF)
        OptimalLow = 1
        OptimalHigh = 1
        MaxVariance = 0
        # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
        Global = np.arange(0, 256)
        GMean = sum(Global * PDF) / CDF[-1]
        for LowT in range(1, 254):
            for HighT in range(LowT + 1, 255):
                try:
                    # Background Intensities Array
                    Back = np.arange(0, LowT)
                    # Low Intensities Array
                    Low = np.arange(LowT, HighT)
                    # High Intensities Array
                    High = np.arange(HighT, 256)
                    # Get Low Intensities CDF
                    CDFL = np.sum(PDF[LowT:HighT])
                    # Get Low Intensities CDF
                    CDFH = np.sum(PDF[HighT:256])
                    # Calculation Mean of Background & The Object (Foreground), Based on CDF & PDF
                    BackMean = sum(Back * PDF[0:LowT]) / CDF[LowT]
                    if CDFL != 0:
                        LowMean = sum(Low * PDF[LowT:HighT]) / CDFL
                    else:
                        LowMean = 0  # or some fallback value
                    HighMean = sum(High * PDF[HighT:256]) / CDFH
                    # Calculate Cross-Class Variance
                    Variance = (CDF[LowT] * (BackMean - GMean) ** 2 + (CDFL * (LowMean - GMean) ** 2) + (
                            CDFH * (HighMean - GMean) ** 2))
                    # Filter Out Max Variance & It's Threshold
                    if Variance > MaxVariance:
                        MaxVariance = Variance
                        OptimalLow = LowT
                        OptimalHigh = HighT
                except RuntimeWarning:
                    pass
        return SpectralThreshold.DoubleThreshold(src, OptimalLow, OptimalHigh, 128, False)
    
    # def spectral_threshold(img):
    #     """Global spectral thresholding on grayscale image."""
    #     hist, _ = np.histogram(img, 256, [0, 256])
    #     mean = np.sum(np.arange(256) * hist) / float(img.size)

    #     optimal_high = 0
    #     optimal_low = 0
    #     max_variance = 0
    #     for high in range(0, 256):
    #         for low in range(0, high):
    #             w0 = np.sum(hist[0:low])
    #             if w0 == 0:
    #                 continue
    #             mean0 = np.sum(np.arange(0, low) * hist[0:low]) / float(w0)
    #             w1 = np.sum(hist[low:high])
    #             if w1 == 0:
    #                 continue
    #             mean1 = np.sum(np.arange(low, high) * hist[low:high]) / float(w1)
    #             w2 = np.sum(hist[high:])
    #             if w2 == 0:
    #                 continue
    #             mean2 = np.sum(np.arange(high, 256) * hist[high:]) / float(w2)
                
    #             variance = w0 * (mean0 - mean) * 2 + w1 * (mean1 - mean) * 2 + w2 * (mean2 - mean) ** 2
    #             if variance > max_variance:
    #                 max_variance = variance
    #                 optimal_high = high
    #                 optimal_low = low

    #     binary = np.zeros(img.shape, dtype=np.uint8)
    #     binary[img < optimal_low] = 0
    #     binary[(img >= optimal_low) & (img < optimal_high)] = 128
    #     binary[img >= optimal_high] = 255
    #     return binary

    @staticmethod
    def LocalThresholding(source: np.ndarray, window_size):
        """
        Applies Local Thresholding To The Given Grayscale Image Using The Given Thresholding Callback Function
        :param source: NumPy Array of The Source Grayscale Image
        :param Regions: Number of Regions To Divide The Image To
        :param ThresholdingFunction: Function That Does The Thresholding
        :return: Thresholded Image
        """
        src = np.copy(source)
        if len(src.shape) > 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        else:
            pass

        YMax, XMax = src.shape
        Result = np.zeros((YMax, XMax))
        YStep = YMax // window_size
        XStep = XMax // window_size
        XRange = []
        YRange = []
        for i in range(0, window_size):
            XRange.append(XStep * i)

        for i in range(0, window_size):
            YRange.append(YStep * i)

        XRange.append(XMax)
        YRange.append(YMax)
        for x in range(0, window_size):
            for y in range(0, window_size):
                Result[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]] = SpectralThreshold.apply_spectral_threshold(src[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]])
        return Result

    @staticmethod
    def DoubleThreshold(Image, LowThreshold, HighThreshold, Weak, isRatio=True):
        """
        Apply Double Thresholding To Image
        :param Image: Image to Threshold
        :param LowThreshold: Low Threshold Ratio/Intensity, Used to Get Lowest Allowed Value
        :param HighThreshold: High Threshold Ratio/Intensity, Used to Get Minimum Value To Be Boosted
        :param Weak: Pixel Value For Pixels Between The Two Thresholds
        :param isRatio: Deal With Given Values as Ratios or Intensities
        :return: Threshold Image
        """

        # Get Threshold Values
        if isRatio:
            High = Image.max() * HighThreshold
            Low = Image.max() * LowThreshold
        else:
            High = HighThreshold
            Low = LowThreshold

        # Create Empty Array
        ThresholdedImage = np.zeros(Image.shape)

        Strong = 255
        # Find Position of Strong & Weak Pixels
        StrongRow, StrongCol = np.where(Image >= High)
        WeakRow, WeakCol = np.where((Image <= High) & (Image >= Low))

        # Apply Thresholding
        ThresholdedImage[StrongRow, StrongCol] = Strong
        ThresholdedImage[WeakRow, WeakCol] = Weak

        return ThresholdedImage


    @staticmethod
    def applySegmentation(img, mode, window_size):
        """Main entry to apply either global or local thresholding."""
        if mode == "Global":
            return SpectralThreshold.apply_spectral_threshold(img)
        elif mode == "Local":
            return SpectralThreshold.LocalThresholding(img, window_size)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'global' or 'local'.")
