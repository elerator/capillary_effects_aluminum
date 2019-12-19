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

        self.img_scaling_ratio = 1#Updated upon resizeEvent
        self.scaled_img_margin_left = None#Updated upon resizeEvent
        self.scaled_img_margin_top = None#Updated upon resizeEvent
        self.original_img_width = None#init via update
        self.original_img_height = None#init via update

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
        scaled = self.pixmap.scaled(size, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation )

        self.scaled_img_margin_left = (self.width() - scaled.width())/2
        self.scaled_img_margin_top = (self.height() - scaled.height())/2
        self.img_scaling_ratio = self.original_img_height/scaled.height()

        #print(self.img_scaling_ratio)

        self.setPixmap(scaled)

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

        if len(frame.shape) == 2:#Grayscale numpy array
            frame = frame - np.min(frame)
            frame = frame / np.max(frame)
            frame = frame * 255
            frame = np.stack((frame,)*3, axis=-1)
            frame = np.array(frame, dtype=np.uint8)
        if frame.shape[2] == 3:
            bytesPerLine = 3 * width
            format = QImage.Format_RGB888
        elif frame.shape[2] == 4:
            #print(frame.shape)
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
        self.original_img_width = width
        self.original_img_height = height
        self.resizeEvent(QResizeEvent(self.size(), QSize()))

def fig2rgb_array(fig):
    """ Converts a matplotlib figure to an rgb array such that it may be displayed as an ImageDisplay
    Args:
        fig: Matplotlib figure
    Returns:
        arr: Image of the plot in the form of a numpy array
    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
