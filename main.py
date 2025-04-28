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
from PyQt5 import QtGui
import k_means 



# Load the UI file
ui, _ = loadUiType("segmentation.ui")

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



        #### segmentation
        self.cluster_method = None
        self.seed_points = []  # Store clicked seeds

        self.cluster_comboBox.currentIndexChanged.connect(self.set_cluster_method)
        self.applyCluster_button.clicked.connect(self.process_clustering)

        self.original_image.mousePressEvent = self.select_seed_point




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
    
    def display_result_on_label(self, label: QLabel, image: np.ndarray):
        """
        Converts a NumPy array to QPixmap and sets it on a QLabel.
        Works with both grayscale and color images.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
            
        # Check if image is grayscale or color
        if len(image.shape) == 2:
            # Grayscale image (2D)
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # Color image (3D)
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
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
            

    #####segmentation 
    def set_cluster_method(self):
        self.cluster_method = self.cluster_comboBox.currentText()

    def select_seed_point(self, event):
        if self.cluster_method != "Region Growing":
            return  # Only allow clicks for Region Growing

        if self.q_image is None:
            return

        # Get the QLabel dimensions
        label_width = self.original_image.width()
        label_height = self.original_image.height()

        # Get the image dimensions
        img_height, img_width = self.img_array.shape[:2]

        # Map clicked coordinates correctly
        clicked_x = event.pos().x()
        clicked_y = event.pos().y()

        # Scale mouse coordinates to image coordinates
        image_x = int(clicked_x * img_width / label_width)
        image_y = int(clicked_y * img_height / label_height)

        # Clip to valid image bounds (in case of slight overflow)
        image_x = np.clip(image_x, 0, img_width - 1)
        image_y = np.clip(image_y, 0, img_height - 1)

        self.seed_points.append((image_y, image_x))  # Note (row, col) == (y, x)

        # Draw red hollow circle on a **copy** of the displayed image
        temp_image = QPixmap(self.q_image)
        painter = QtGui.QPainter(temp_image)
        painter.setPen(QtGui.QPen(Qt.red, 3))
        scale_x = label_width / img_width
        scale_y = label_height / img_height
        painter.drawEllipse(int(image_x * scale_x), int(image_y * scale_y), 8, 8)
        painter.end()

        self.original_image.setPixmap(temp_image)

    def process_clustering(self):
                # Get the current method (in case it wasn't set yet)
        if self.cluster_method is None:
            self.cluster_method = self.cluster_comboBox.currentText()
        if self.cluster_method == "Region Growing":
            import regiongrowing

            if not self.seed_points:
                print("No seed points selected!")
                return

            threshold = int(self.threshold_lineEdit.text())

            result = regiongrowing.apply_region_growing(self.img_array, self.seed_points, threshold)

            self.display_result_on_label(self.result_image, result)

        elif self.cluster_method == "Agglomerative Clustering":
            # import agglomerative
            pass

        elif self.cluster_method == "K-Means":
            # Get parameters from combo boxes and line edits
            num_clusters = self.cluster_lineEdit.text()
            threshold = self.threshold_lineEdit.text()
            
            # Apply k-means clustering
            result = k_means.apply_kmeans(self.img_array, num_clusters, threshold)
            
            # Display result
            self.display_result_on_label(self.result_image, result)

        elif self.cluster_method == "Mean-Shift":
            pass

        else:
            raise ValueError(f"Unknown segmentation method: '{self.cluster_method}'")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())