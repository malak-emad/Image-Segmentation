import sys
import os
import argparse
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
                            QTabWidget, QTextEdit, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import logging
logger = logging.getLogger(__name__)
from spectral_threshold import SpectralThreshold
import pyqtgraph as pg


# Load the UI file
ui, _ = loadUiType("Image-Segmentation\segmentation.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        self.threshold_method = None
        self.threshold_mode = None

        self.upload_button.clicked.connect(self.uploadImage)
        self.threshold_comboBox.currentIndexChanged.connect(self.set_threshold_method)
        self.local_radioButton.clicked.connect(lambda: self.set_threshold_mode("Local"))
        self.global_radioButton.clicked.connect(lambda: self.set_threshold_mode("Global"))
        self.applyThreshold_button.clicked.connect(self.process_threshold)
        self.seg_frame.hide()



    def uploadImage(self, value):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.webp);;All Files (*)", options=options)
            
            if file_path:
                self.value = value
                self.q_image, self.image = self.process_and_store_image(file_path)  
                self.original_image.setPixmap(QPixmap.fromImage(self.q_image))
                self.result_image.setPixmap(QPixmap())

                #set scaled contents for each QLabel only once
                self.original_image.setScaledContents(True)
                self.result_image.setScaledContents(True)
            print("upload")

    def process_and_store_image(self, file_path):
        original_image = Image.open(file_path).convert("RGB")
        self.img_array = np.array(original_image)
    
        #convert PIL image to QImage
        height, width, channels = self.img_array.shape
        bytes_per_line = channels * width
        q_image = QImage(self.img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        return q_image, self.img_array
    
    # Function to display the output image on its label
    def display_result_on_label(self, label: QLabel, image: np.ndarray):
        """
        Converts a grayscale NumPy array to QPixmap and sets it on a QLabel.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        height, width = image.shape
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True) 
    
    def set_threshold_method(self):
        self.threshold_method = self.threshold_comboBox.currentText()
    
    def set_threshold_mode(self, mode):
        self.threshold_mode = mode
        if mode == "Local":
            self.seg_frame.show()
        else: 
            self.seg_frame.hide()

    def process_threshold(self):
        match self.threshold_method:
            case "Optimal Thresholding":
                pass
            case "OTSU Thresholding":
                pass
            case "Spectral Thresholding":
                gray = cv2.cvtColor(self.img_array, cv2.COLOR_RGB2GRAY)
                if self.threshold_mode == "Global":
                    self.window_size = 1
                else: 
                    self.window_size = int(self.wSize_lineEdit.text())
                segmented_image = SpectralThreshold.applySegmentation(gray, self.threshold_mode, self.window_size)
                self.display_result_on_label(self.result_image, segmented_image)

            case _:
                raise ValueError(f"Unknown thresholding method: '{self.threshold_method}'")
            


    


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())