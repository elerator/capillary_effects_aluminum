from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np


import imageio
import os

class ImageDisplay(QLabel):
    """ Displays an image as a QLabel
        Drawing is realized such that the aspect-ratio is kept constant
        and the image fills up all the available space in the layout"""
    def __init__(self, video, parent=None, centered = True):
        super(ImageDisplay, self).__init__(parent)
        #set background to black and border to 0
        self.setStyleSheet("background-color: rgb(0,0,0); margin:0px; border:0px solid rgb(0, 255, 0); ")
        self.setMinimumSize(320, 180)#Set minimum size
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)# Set size policy to expanding
        self.setAlignment(Qt.AlignCenter)
        self.update()
        self.set_light_gray_background()

    def set_white_background(self):
        self.setStyleSheet("background-color: rgb(255,255,255); margin:0px; border:0px solid rgb(0, 255, 0); ")

    def set_light_gray_background(self):
        self.setStyleSheet("background-color: rgb(239,240,241); margin:0px; border:0px solid rgb(0, 255, 0); ")

    def set_gray_background(self):
        self.setStyleSheet("background-color: rgb(236,232,228); margin:0px; border:0px solid rgb(0, 255, 0); ")

    def resizeEvent(self, event):
        """ Rescales the Pixmap that contains the image when QLabel changes size
            Args:
                event: QEvent
        """
        size = self.size()
        size = QSize(int(size.width()),int(size.height()))
        scaledPix = self.pixmap.scaled(size, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation )
        self.setPixmap(scaledPix)

    def update(self, frame = None):
        """ Upates the pixmap when a new frame is to be displayed.
            Args:
                frame: The frame to update
        """
        if type(frame) == type(None):#Init blank frame if no video is set yet
            frame = np.ndarray((9,16,3), dtype = np.byte)
            frame.fill(100)

        height = frame.shape[0]
        width = frame.shape[1]
        bytesPerLine = None
        format = None

        if frame.shape[2] == 3:
            bytesPerLine = 3 * width
            format = QImage.Format_RGB888
        elif frame.shape[2] == 4:
            print(frame.shape)
            bytesPerLine = 4 * width
            format = QImage.Format_ARGB32
        else:
            print("Image format not supported")
            return

        image = QImage(frame.data, width, height, bytesPerLine, format)
        self.pixmap = QPixmap(image)
        size = self.size()
        scaledPix = self.pixmap.scaled(size, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation)
        self.setPixmap(scaledPix)
