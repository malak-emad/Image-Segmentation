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
from optimal_threshold import OptimalThreshold
import pyqtgraph as pg
from PyQt5 import QtGui
import k_means
import mean_shift



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
        Converts a NumPy array (grayscale or RGB) to QPixmap and sets it on a QLabel.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Handle grayscale and color separately
        if len(image.shape) == 2:  # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # Color
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            raise ValueError("Unsupported image format for displaying.")

        pixmap = QPixmap.fromImage(q_image)

        # Resize pixmap nicely
        pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

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
                gray = cv2.cvtColor(self.img_array, cv2.COLOR_RGB2GRAY)
                if self.threshold_mode == "Global":
                    self.window_size = 1
                else: 
                    self.window_size = int(self.wSize_lineEdit.text())
                
                segmented_image = OptimalThreshold.applySegmentation(gray, self.threshold_mode, self.window_size)
                self.display_result_on_label(self.result_image, segmented_image)
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
        if self.cluster_method == "Mean-Shift":
            self.label_6.setText("Window Size:")
        else: 
            self.label_6.setText("Clusters:")

    def select_seed_point(self, event):
        if self.cluster_method != "Region Growing":
            return

        if self.q_image is None:
            return

        # QLabel (frame) size
        label_width = self.original_image.width()
        label_height = self.original_image.height()

        # Original image size
        img_height, img_width = self.img_array.shape[:2]

        # Pixmap size (displayed image inside QLabel)
        pixmap = self.original_image.pixmap()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Mouse click position relative to QLabel
        clicked_x = event.pos().x()
        clicked_y = event.pos().y()

        # Rescale click to pixmap coordinates
        clicked_x_on_pixmap = clicked_x * pixmap_width / label_width
        clicked_y_on_pixmap = clicked_y * pixmap_height / label_height

        # Now scale pixmap coordinates back to real image coordinates
        image_x = int(clicked_x_on_pixmap * img_width / pixmap_width)
        image_y = int(clicked_y_on_pixmap * img_height / pixmap_height)

        # Clip coordinates to image
        image_x = np.clip(image_x, 0, img_width - 1)
        image_y = np.clip(image_y, 0, img_height - 1)

        # Save the real seed pixel for algorithm
        self.seed_points.append((image_y, image_x))

        # Draw circle centered on correct place
        temp_pixmap = pixmap.copy()
        painter = QtGui.QPainter(temp_pixmap)
        painter.setPen(QtGui.QPen(Qt.red, 2))
        radius = 5
        painter.drawEllipse(int(clicked_x_on_pixmap) - radius, int(clicked_y_on_pixmap) - radius, 2 * radius, 2 * radius)
        painter.end()

        self.original_image.setPixmap(temp_pixmap)

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
                import agglomerative

                num_clusters = int(self.cluster_lineEdit.text())
                result = agglomerative.apply_agglomerative_clustering(self.img_array, num_clusters)

                self.display_result_on_label(self.result_image, result)
                    

            elif self.cluster_method == "K-Means":
                # Get parameters from combo boxes and line edits
                num_clusters = self.cluster_lineEdit.text()
                threshold = self.threshold_lineEdit.text()
                
                # Apply k-means clustering
                result = k_means.apply_kmeans(self.img_array, num_clusters, threshold)
                
                # Display result
                self.display_result_on_label(self.result_image, result)

            elif self.cluster_method == "Mean-Shift":
                # Get parameters from line edits
                window_size = self.cluster_lineEdit.text()
                convergence_threshold = self.threshold_lineEdit.text()
                
                # Show progress message
                print("Starting Mean-Shift segmentation. This may take a while...")
                
                # Apply mean-shift clustering
                result = mean_shift.apply_meanshift(self.img_array, window_size, convergence_threshold)
                
                # Display result
                self.display_result_on_label(self.result_image, result)
                print("Mean-Shift segmentation completed!")

            else:
                raise ValueError(f"Unknown segmentation method: '{self.cluster_method}'")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())