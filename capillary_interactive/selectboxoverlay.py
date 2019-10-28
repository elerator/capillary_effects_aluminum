
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import numpy as np
import sip


class SelectBoxOverlay(QWidget):
  coordinates = pyqtSignal(QRect)

  def __init__(self, w, parent = None):
      #Make sure that the parent widget of the imagedisplay has no border
      QWidget.__init__(self, w)
      self.w = w

      self.begin = QtCore.QPoint(0,0)
      self.end = QtCore.QPoint(0,0)

      self.box_begin = QtCore.QPoint(0,0)#For the final selection
      self.box_end = QtCore.QPoint(0,0)

      self.currently_drawing = False

      self.setGeometry(w.geometry())
      self.temporarily_disabled = False
      self.update()

  def toggle_enabled(self):
      self.temporarily_disabled = not self.temporarily_disabled

  def get_box_coordinates(self):
      """ returns the coordinates of the current selection"""
      return QRect(self.box_begin,self.box_end)

  def paintEvent(self, event):
      if self.temporarily_disabled:
          return
      self.setGeometry(self.w.geometry())
      if self.currently_drawing:
          self.draw_rect()

  def draw_rect(self):
      qp = QPainter(self)
      br = QBrush(QColor(100, 100, 100, 100))
      qp.setBrush(br)
      qp.drawRect(QRect(QtCore.QPoint(self.begin.x(),self.begin.y()),
                        QtCore.QPoint(self.end.x(),self.end.y())))

  def mousePressEvent(self, event):
      """ Resets coordinates of the select box. Sets beginning point to mouse pos.
          Args:
              event: GUI event
      """
      self.currently_drawing = True
      self.begin =  event.pos()
      self.end =  event.pos()
      self.update()

  def mouseMoveEvent(self, event):
      """ Sets end point to mouse pos. Updates the select_box overlay.
          Args:
              event: GUI event
      """
      self.end = event.pos()
      self.update()

  def mouseReleaseEvent(self, event):
      """ Copies the current coordinates to respective attributes.
          If permanent_show is set to false, deletes select_box view.
          Args:
              event: GUI event
      """
      self.currently_drawing = False

      self.box_begin = self.begin
      self.box_end = event.pos()
      self.begin = self.begin
      self.end = event.pos()

      self.coordinates.emit(QRect(self.get_box_coordinates().topLeft(),self.get_box_coordinates().bottomRight()))
      self.update()
