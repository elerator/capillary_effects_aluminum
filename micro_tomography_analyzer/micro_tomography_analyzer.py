from micro_tomography_analyzer_ui import Ui_MicroTomographyAnalyzer

import breeze_resources#light layout
import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QButtonGroup
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QFile, QTextStream

import os
from PIL import Image
import PIL

import sys
from time import sleep

import traceback

from skimage.filters import rank
from skimage.morphology import disk

from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering

from fast_density_clustering import FastDensityClustering
from imagedisplay import fig2rgb_array


import matplotlib.pyplot as plt
from pandas_model import PandasModel

from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

class FileDialog(QWidget):
    outfilepath = pyqtSignal(str)
    folder = pyqtSignal(str)
    filepath = pyqtSignal(str)

    def __init__(self, file_ending = ".csv"):
        self.file_ending = file_ending
        self.columns = None
        QWidget.__init__(self)

    def create_output_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(None,"Select the output file", "", self.file_ending[:] +" (*."+self.file_ending[1:]+");;", options=options)

        if filename:
            if not filename.endswith(self.file_ending):
                filename += self.file_ending
            self.outfilepath.emit(filename)
        return filename

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(None,"Select the output file", "", self.file_ending[1:] +" (*."+self.file_ending[1:]+");;", options=options)
        self.outfilepath.emit(filename)
        return filename


    def get_folder_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(None,"Select folder...",os.getcwd(),options=options)
        if path:
            self.folder.emit(path)
        return path

class MicroTomographyAnalyzer(QWidget):
    def __init__(self, widget_handled, ui):
        """ Initializes Main App.
        args:
            widget_handled: For this widget events are handeled by MainApp
            ui: User interface
        """
        QWidget.__init__(self, widget_handled)

        self.ui = ui
        self.widget_handled = widget_handled #Widget for event Handling. All events are checked and if not arrow keys passed on. See eventFilter below
        self.source_dir = ""
        self.tensor_loader = MicroTomographyAnalyzer.TensorLoader()
        self.tensor = None

        self.slicer = MicroTomographyAnalyzer.Slicer(self)
        self.rotator = MicroTomographyAnalyzer.Rotator(self)
        self.columnwise_polymer = MicroTomographyAnalyzer.ColumnwisePolymer(self)

        self.column_detector = MicroTomographyAnalyzer.ColumnDetector(self)
        self.imbibition_front = MicroTomographyAnalyzer.ImbibitionFront(self)

        self.slice_xy = True
        self.slice_xz = False
        self.slice_yz = False
        self.current_slice = 0
        self.make_connections()

    def set_display_axis(self, axis):
        if axis == "yx":
            self.slice_xy = True
            self.slice_xz = False
            self.slice_yz = False
        elif axis == "yz":
            self.slice_xy = False
            self.slice_xz = True
            self.slice_yz = False
        elif axis == "xz":
            self.slice_xy = False
            self.slice_xz = False
            self.slice_yz = True
        self.display_slice(0)

    def make_connections(self):
        self.tensor_loader.tensor.connect(self.set_tensor)
        self.tensor_loader.update_info.connect(self.ui.progress.setValue)
        self.ui.next.clicked.connect(lambda: self.display_slice(self.current_slice+1))
        self.ui.previous.clicked.connect(lambda: self.display_slice(self.current_slice-1))
        self.ui.axis.currentTextChanged.connect(self.set_display_axis)
        self.slicer.output_tensor.connect(self.set_tensor)
        self.ui.actionRotate.triggered.connect(lambda x: self.ui.tools.setCurrentWidget(self.ui.page_rotating))
        self.ui.actionSlice_tensor.triggered.connect(lambda x: self.ui.tools.setCurrentWidget(self.ui.page_slicing))
        self.ui.actionDetect.triggered.connect(lambda x: self.ui.tools.setCurrentWidget(self.ui.page_column_detection))
        self.ui.actionImbibition.triggered.connect(lambda x: self.ui.tools.setCurrentWidget(self.ui.page_imbibition_front))
        self.ui.next_display_page.clicked.connect(self.next_display_page)
        self.ui.previous_display_page.clicked.connect(self.previous_display_page)

        self.column_detector.column_positions.connect(self.imbibition_front.set_column_positions)
        self.column_detector.column_positions.connect(self.columnwise_polymer.set_column_positions)


    def next_display_page(self):
        if self.ui.display_stack.currentIndex()+1 == self.ui.display_stack.count():
            return
        try:
            self.ui.display_stack.setCurrentIndex(self.ui.display_stack.currentIndex()+1)
        except:
            pass

    def previous_display_page(self):
        if self.ui.display_stack.currentIndex() == 0:
            return
        try:
            self.ui.display_stack.setCurrentIndex(self.ui.display_stack.currentIndex()-1)
        except:
            pass

    def set_source_dir(self, source_dir):
        self.source_dir = source_dir
        files = [os.path.join(self.source_dir,x) for x in os.listdir(self.source_dir)]
        self.ui.loading_info_stack.setCurrentIndex(1)
        self.tensor_loader.set_files(files)
        self.tensor_loader.start()

    def set_tensor(self, tensor):
        self.tensor = tensor
        self.ui.loading_info_stack.setCurrentIndex(0)
        self.ui.file_info.setText("Loaded tensor successfully")
        self.display_slice(self.current_slice)
        self.slicer.update_axis_bounds()

    def get_tensor(self):
        return self.tensor

    def get_current_slice(self):
        slice = None
        idx = self.current_slice
        if self.slice_xy:
            slice = self.tensor[:,:,idx].copy()#remove copy in this line?
        elif self.slice_xz:
            slice = self.tensor[idx,:,:].copy()#TODO: Check if correct...
        elif self.slice_yz:
            slice = self.tensor[:,idx,:].copy()
        return slice

    def display_slice(self, idx):
        old_idx = self.current_slice
        try:
            self.current_slice = idx
            self.ui.slices.update(self.get_current_slice())
            self.ui.slice.setValue(self.current_slice)
        except Exception as err:
            self.current_slice = old_idx
            print(err)
            traceback.print_tb(err.__traceback__)

    def set_column_positions(self, columns):
        self.set_column_positions = columns

    def eventFilter(self, source, event):
        """ Filters key events such that arrow keys may be handled.
            Args:
                source: Source of event
                event: Event to be handled
        """
        if event.type() == QtCore.QEvent.KeyRelease:
            id_right = 16777236
            id_left = 16777234
            if event.key() == id_right:
                self.display_slice(self.current_slice+1)

            elif event.key() == id_left:
                self.display_slice(self.current_slice-1)
        try:#When closing the app the widget handled might already have been destroyed
            return True#self.widget_handled.eventFilter(source, event)#Execute the default actions for the event
        except:
            return True#a true value prevents the event from being sent on to other objects

    class TensorLoader(QThread):#TODO. Set deamom
        tensor = pyqtSignal(np.ndarray)
        update_info = pyqtSignal(int)

        def __init__(self):
            super(MicroTomographyAnalyzer.TensorLoader, self).__init__()
            self.files = None

        def set_files(self,files):
            self.files = files

        def run(self):
            if not self.files:
                return
            files = self.files
            try:
                im1 = Image.open(files[0])
                arr1 = np.array(im1)
                shape = [len(files), arr1.shape[0],arr1.shape[1]]
                im1.close()
                arr1 = None

                tensor = np.ndarray(shape=shape, dtype=np.uint16)
                for i, x in enumerate(files):
                    im = Image.open(x)
                    tensor[i] = np.array(im, dtype=np.uint16)
                    im.close()
                    self.update_info.emit(int(100*(i/len(files))))
                tensor = np.einsum('zyx->yxz', tensor)
                self.tensor.emit(tensor)
            except Exception as e:
                print("No valid folder. Loading tensor failed")
                print(e)

    class Slicer(QWidget):
        output_tensor = pyqtSignal(np.ndarray)
        def __init__(self, parent_controller):
            super(MicroTomographyAnalyzer.Slicer,self).__init__()
            self.ui = parent_controller.ui
            self.parent_controller = parent_controller
            self.make_connections()

        def reset_current_slice(self):
            """ Resets currently displayed slice """
            try:
                slice = self.parent_controller.get_current_slice()
                self.parent_controller.slices.update(slice)
            except Exception as err:
                traceback.print_tb(err.__traceback__)

        def update_axis_bounds(self):
            tensor = self.parent_controller.get_tensor()
            if type(tensor)==type(None):
                return
            axis = self.ui.axis.currentText()
            low_dim1 = self.ui.low_bound_dim1
            low_dim2 = self.ui.low_bound_dim2
            high_dim1 = self.ui.high_bound_dim1
            high_dim2 = self.ui.high_bound_dim2
            low_dim1.setValue(0)
            low_dim2.setValue(0)
            low_dim1.setMinimum(0)
            low_dim2.setMinimum(0)
            shape = None
            if axis == "yx":
                slice = tensor[:,:,0]
                shape = slice.shape
            elif axis == "yz":
                slice = tensor[:,0,:]
                shape = slice.shape
            elif axis == "xz":
                slice = tensor[0,:,:]
                shape = slice.shape
            low_dim1.setMaximum(shape[0])
            low_dim2.setMaximum(shape[1])

            high_dim1.setMaximum(shape[0])
            high_dim2.setMaximum(shape[1])
            high_dim1.setValue(shape[0])
            high_dim2.setValue(shape[1])

        def reset_preview(self):
            self.update_axis_bounds()
            self.slice_preview()

        def slice_preview(self):
            try:
                slice = self.parent_controller.get_current_slice()
                low_dim1 = self.ui.low_bound_dim1.value()
                low_dim2 = self.ui.low_bound_dim2.value()
                high_dim1 = self.ui.high_bound_dim1.value()
                high_dim2 = self.ui.high_bound_dim2.value()
                slice = slice[low_dim1:high_dim1,low_dim2:high_dim2]
                self.parent_controller.ui.slices.update(slice)
            except Exception as err:
                #traceback.print_tb(err.__traceback__)
                pass

        def make_connections(self):
            self.parent_controller.ui.axis.currentTextChanged.connect(lambda x: self.update_axis_bounds())

            self.parent_controller.ui.low_bound_dim1.valueChanged.connect(lambda x: self.slice_preview())
            self.parent_controller.ui.low_bound_dim2.valueChanged.connect(lambda x: self.slice_preview())
            self.parent_controller.ui.high_bound_dim1.valueChanged.connect(lambda x: self.slice_preview())
            self.parent_controller.ui.high_bound_dim2.valueChanged.connect(lambda x: self.slice_preview())
            self.parent_controller.ui.reset_slicing.clicked.connect(lambda x: self.reset_preview())
            self.parent_controller.ui.apply_slicing.clicked.connect(lambda x: self.apply_slicing())

        def apply_slicing(self):
            message = "Note that you cannot undo this step but you must load the tensor from file again to go back."
            message += "\n\nDo you really want to overwrite the tensor with the current subtensor?"
            response = QMessageBox.question(self, 'Warning', message,QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if response == QMessageBox.No:
                return
            try:
                tensor = self.parent_controller.tensor
                low_dim1 = self.ui.low_bound_dim1.value()
                low_dim2 = self.ui.low_bound_dim2.value()
                high_dim1 = self.ui.high_bound_dim1.value()
                high_dim2 = self.ui.high_bound_dim2.value()
                if self.parent_controller.slice_xy:
                    tensor = tensor[low_dim1:high_dim1,low_dim2:high_dim2,:]
                elif self.parent_controller.slice_yz:
                    tensor = tensor[low_dim1:high_dim1,:,low_dim2:high_dim2]
                elif self.parent_controller.slice_xz:
                    tensor = tensor[:,low_dim1:high_dim1,low_dim2:high_dim2]
                self.output_tensor.emit(tensor)
            except Exception as e:
                print(e)
                traceback.print_tb(e.__traceback__)

    class Rotator(QThread):
        output_tensor = pyqtSignal(np.ndarray)
        progress = pyqtSignal(int)
        done = pyqtSignal(int)
        def __init__(self,outer):
            super(MicroTomographyAnalyzer.Rotator,self).__init__()
            self.ui = outer.ui
            self.outer = outer
            self.angle = None
            self.make_connections()
        def make_connections(self):
            self.ui.rotation_angle.valueChanged.connect(self.rotate_preview)
            self.output_tensor.connect(self.outer.set_tensor)
            self.progress.connect(self.ui.progress.setValue)
            self.done.connect(lambda: self.ui.loading_info_stack.setCurrentIndex(0))
            self.ui.apply_rotation.clicked.connect(self.rotate_tensor)

        def rotate_preview(self, angle):
            self.angle = angle
            slice = self.outer.get_current_slice()
            slice = Image.fromarray(np.array(slice,dtype=np.uint32)).rotate(angle,resample=PIL.Image.BICUBIC)
            self.ui.slices.update(np.array(slice))

        def rotate_tensor(self):
            message = "Note that you cannot undo this step but you must load the tensor from file again to go back."
            message += "Rotation is lossless only for (+/-) 90, 180 and 270 degrees"
            message += "\n\nDo you really want to overwrite the tensor with the specified rotation in the current dimension?"
            response = QMessageBox.question(None, 'Warning', message,QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if response == QMessageBox.No:
                return
            self.ui.loading_info_stack.setCurrentIndex(1)
            if not self.angle:
                return
            self.start()

        def run(self):
            angle = self.angle
            tensor = self.outer.tensor
            output_tensor = np.ndarray(shape=tensor.shape)
            n_rotations = 0
            if self.outer.slice_xy:
                n_rotations = tensor.shape[2]
                for i in range(n_rotations):
                    self.progress.emit(int((i/n_rotations)*100))
                    output_tensor[:,:,i] = np.array(Image.fromarray(np.array(tensor[:,:,i],dtype=np.uint32)).rotate(angle,resample=PIL.Image.BICUBIC))
            elif self.outer.slice_xz:
                n_rotations = tensor.shape[0]
                for i in range(n_rotations):
                    self.progress.emit(int((i/n_rotations)*100))
                    output_tensor[i,:,:] = np.array(Image.fromarray(np.array(tensor[i,:,:],dtype=np.uint32)).rotate(angle,resample=PIL.Image.BICUBIC))
            elif self.outer.slice_yz:
                n_rotations = tensor.shape[1]
                for i in range(n_rotations):
                    self.progress.emit(int((i/n_rotations)*100))
                    output_tensor[:,i,:] = np.array(Image.fromarray(np.array(tensor[:,i,:],dtype=np.uint32)).rotate(angle,resample=PIL.Image.BICUBIC))
            self.output_tensor.emit(output_tensor)
            self.done.emit(1)

    class ColumnDetector(QThread):
        column_positions = pyqtSignal(list)#List of two lists: X and Y coordinates
        plot = pyqtSignal(np.ndarray)
        points_sampled = pyqtSignal(int)
        def __init__(self, outer):
            super(MicroTomographyAnalyzer.ColumnDetector,self).__init__()
            self.outer = outer
            self.ui = outer.ui
            self.z_for_mean_start = 0
            self.z_for_mean_end = 0
            self.threshold = 70
            self.clustering_choice = QButtonGroup()
            self.clustering_choice.addButton(self.ui.agglomerative_clustering)
            self.clustering_choice.addButton(self.ui.custom_clustering)
            self.ui.custom_clustering.setChecked(True)

            self.make_connections()

        def make_connections(self):
            self.ui.start_column_detection.clicked.connect(self.detect_columns)
            self.column_positions.connect(self.outer.set_column_positions)
            self.plot.connect(lambda x: (self.ui.display_stack.setCurrentWidget(self.ui.page_detected_columns),self.ui.detected_columns.update(x)))
            self.points_sampled.connect(lambda x: self.ui.n_points_sampled.setText(str(x)))

        def detect_columns(self):
            self.selem_size = self.ui.selem_size.value()
            self.z_for_mean_start = self.ui.z_for_mean_start.value()
            self.z_for_mean_end = self.ui.z_for_mean_end.value()
            self.threshold = self.ui.threshold_column_detection.value()

            self.start()

        def print_points_and_background(self, img, x,y, point_size=.3, marker ="."):
            fig, ax = plt.subplots(1, figsize=(12,10))

            ax.set_xlim((0, img.shape[1]))
            ax.set_ylim((img.shape[0], 0))
            ax.imshow(img,cmap="gray")
            ax.scatter(x,y, s=point_size,c="red",marker=marker)
            return fig

        def run(self):
            tensor = self.outer.get_tensor()
            if type(tensor) == type(None):
                return

            if self.z_for_mean_start>=self.z_for_mean_end:
                return

            # 1. Select slices normalize, compute average along axiz z and perform local histogram equalization
            subtensor = tensor[:,:,self.z_for_mean_start:self.z_for_mean_end]
            img = np.mean(subtensor, axis =2)
            img -= np.min(img)
            img /= np.max(img)

            img = rank.equalize(img, selem=disk(self.selem_size))
            img = gaussian_filter(img,1)

            img = np.mean(subtensor, axis =2)
            img -= np.min(img)
            img /= np.max(img)

            # 2. Thresholding
            binarized = img<self.threshold/100
            y,x = np.where(binarized)
            self.points_sampled.emit(len(y))

            fig = self.print_points_and_background(img,x,y)
            fig = fig2rgb_array(fig)
            self.plot.emit(fig)

            # End processing at this point if the user wants the preview only
            if self.ui.column_detection_preview_only.isChecked():
                return

            # 3. Clustering
            data = np.array(list(zip(y,x)))
            if len(data) == 0:
                print("No points sampled")
                return

            centroid_x = []
            centroid_y = []

            if self.ui.agglomerative_clustering.isChecked():
                try:
                    if len(y)>20000:
                        raise Exception("Too many samples")
                    cluster = AgglomerativeClustering(n_clusters=None,distance_threshold=5, linkage='complete')
                    labels = cluster.fit_predict(data)

                    print("clustering finished")
                    centroid_x = []
                    centroid_y = []
                    for i in set(labels):
                        samples_of_cluster = labels==i
                        centroid_x.append(np.mean(x[samples_of_cluster]))
                        centroid_y.append(np.mean(y[samples_of_cluster]))
                except Exception as e:
                    QMessageBox.information(None, "Clustering failed","Could not detect columns likely because you sampled too many values (Out of memory error)")
            elif self.ui.custom_clustering.isChecked():
                print("Starting clustering custom")
                centroid_y, centroid_x, _, _ = FastDensityClustering.density_clustering(binarized,None,"uniform",4)

            self.column_positions.emit([centroid_x,centroid_y])
            fig = self.print_points_and_background(img,centroid_x,centroid_y,point_size=20)
            fig = fig2rgb_array(fig)
            self.plot.emit(fig)

    class ColumnwisePolymer(QThread):
        """ Uses grayvalue slope to detect likely position of start, end and length of infiltrated part in each column"""
        measurements = pyqtSignal(pd.DataFrame)
        def __init__(self, outer):
            super(MicroTomographyAnalyzer.ColumnwisePolymer, self).__init__()
            self.outer = outer
            self.ui = outer.ui
            self.imbibition_front = None
            self.table_model = None
            self.outfile_specifier = FileDialog(".csv")

            self.column_x = None
            self.column_y = None

            self.make_connections()

        def make_connections(self):
            self.ui.start_sampling.clicked.connect(self.start)
            self.measurements.connect(self.set_model)
            self.ui.save_polymer_level.clicked.connect(self.outfile_specifier.create_output_file)
            self.outfile_specifier.outfilepath.connect(self.save_csv)
            self.ui.update_imbibiton_front.clicked.connect(self.update_figure)
            self.ui.save_grayvalue_slopes.clicked.connect(self.save_grayvalue_slopes)

        def set_column_positions(self, column_positions):
            self.column_x = column_positions[0]
            self.column_y = column_positions[1]

        def save_grayvalue_slopes(self):
            try:
                file_dialog = FileDialog(".png")
                filepath = file_dialog.create_output_file()

                polymer_level_columnwise = np.array(self.imbibition_front,dtype=np.float32)
                for i,x in enumerate(polymer_level_columnwise):
                    x -= np.min(x)
                    x /= np.max(x)
                    polymer_level_columnwise[i] = x

                polymer_level_columnwise = np.array(polymer_level_columnwise*255, dtype=np.uint8)

                img = Image.fromarray(polymer_level_columnwise)
                img.save(filepath)
            except Exception as e:
                print(e)

        def set_polymer_level(self, imbibition_front):
            self.imbibition_front = imbibition_front
            self.display(imbibition_front)

        def sample(self):
            if type(self.imbibition_front) == type(None):
                QMessageBox.information(None, "Cuould not find imbibition front","Perform sampling of grayvalue slopes first.")
                return
            self.start()

        def minima(self, vector, pre_smoothing=2):
            return argrelextrema(gaussian_filter(vector,pre_smoothing),np.less)[0]

        def maxima(self, vector, pre_smoothing=2):
            """ Returns indices of local maxima sorted by value at each indice starting with the highest value"""
            return argrelextrema(gaussian_filter(vector,pre_smoothing),np.greater)[0]

        def minima(self, vector, pre_smoothing=2):
            return argrelextrema(gaussian_filter(vector,pre_smoothing),np.less)[0]

        def maxima(self, vector, pre_smoothing=2):
            return argrelextrema(gaussian_filter(vector,pre_smoothing),np.greater)[0]

        def sort_by(self, to_sort,order_by):
            return np.array(sorted(zip(order_by,to_sort)))

        def figure(self, polymer_level_columnwise, x1=None, x2=None):
            """ Plot a figure where each row refers to the grayvalue slope of one pore.
                Values for each row are standardized by substracting the mean and deviding by the maximal value.
                Expected values for the imbibition front are determined by taking the first two local minima of the mean along y axis if x1 or x2 is None.
                This is done as the imbibition front is dark and by averaging one looks for vertical lines where the polymer level or polymer/aluminum interface lines up.
                Named expected values are plotted as red lines.
            Args:
                polymer_level_columnwise: A matrix containing the grayvalue slope
                x1: Expected value of the left front. Is plotted as a red vertical line.
                x2: Expected value of the right front. Is plitted as red vertical line.
            """

            if not x1 or not x2:
                print("Get E[x]")
                try:
                    #Get largest local maxima -> Expected value for column
                    mean = np.mean(polymer_level_columnwise, axis = 0)
                    peaks = self.maxima(mean)
                    vals_at_peaks = mean[peaks]
                    x1, x2 = self.sort_by(peaks,vals_at_peaks)[-2:,1]#Get two largest peaks
                except:
                    pass
                if not x1:
                    x1 = 0
                if not x2:
                    x2 = 0

            self.ui.left_expected_value.setValue(x1)
            self.ui.right_expected_value.setValue(x2)

            standardized_front = np.array(polymer_level_columnwise, dtype=np.float32)
            for i,x in enumerate(standardized_front):
                x -= np.min(x)
                x /= np.max(x)
                standardized_front[i] = x

            fig, ax = plt.subplots(1, figsize=(12,10))
            ax.axvline(x1, color='r', linestyle='-')
            ax.axvline(x2, color='r', linestyle='-')
            ax.imshow(standardized_front,aspect=0.1, cmap="gray")
            return fig

        def display(self, polymer_level_columnwise):
            self.ui.imbibition_front.update(fig2rgb_array(self.figure(polymer_level_columnwise)))
            self.ui.display_stack.setCurrentWidget(self.ui.page_front_display)

        def update_figure(self):
            """ Updates figure if imbibition front was set already. Uses the expected value from the respective spin boxes in the ui."""
            if type(self.imbibition_front) == type(None):
                print("No imbibition_front")
                return
            self.ui.imbibition_front.update(fig2rgb_array(self.figure(self.imbibition_front, self.ui.left_expected_value.value(), self.ui.right_expected_value.value())))

        def minimum_within_radius(self, vector, expected_value, radius):
            """ Returns minimum within radius """
            window = vector[expected_value-radius:expected_value+radius]
            return expected_value-radius+np.median(np.where(window==np.min(window)))

        def minima(self, vector, pre_smoothing=2):
            return argrelextrema(gaussian_filter(vector,pre_smoothing),np.less)[0]

        def closest_substantial_minimum(self, vector, expected_value, radius, pre_smoothing=2):
            window = vector[expected_value-radius:expected_value+radius]
            mins = self.minima(vector, pre_smoothing)
            if len(mins) ==0:
                return [np.nan]
            idx = np.argmin([np.abs(x-expected_value) for x in mins])
            return mins[idx]

        def run(self):
            #Get expected values
            polymer_level_columnwise = self.imbibition_front
            mean = np.mean(polymer_level_columnwise, axis = 0)
            peaks = self.maxima(mean)
            vals_at_peaks = mean[peaks]
            x1, x2 = self.sort_by(peaks,vals_at_peaks)[-2:,1]#Get two largest peaks
            if not x1:
                x1 = 0
            if not x2:
                x2 = 0
            x1 = int(x1)
            x2 = int(x2)

            sampling_rule = str(self.ui.sampling_rule.currentText())
            try:
                left_front = []
                right_front = []
                for row in polymer_level_columnwise:
                        if sampling_rule == "Minimum within region of interest":
                            left_front.append(self.minimum_within_radius(row,x1,self.ui.maxdist_extrema.value()))
                            right_front.append(self.minimum_within_radius(row,x2,self.ui.maxdist_extrema.value()))
                        elif sampling_rule ==  "Closest substantial minimum to expected value":
                            left_front.append(self.closest_substantial_minimum(row,x1,self.ui.maxdist_extrema.value()))
                            right_front.append(self.closest_substantial_minimum(row,x2,self.ui.maxdist_extrema.value()))
            except Exception as e:
                traceback.print_tb(e.__traceback__)


            left_front = np.array(left_front)
            right_front = np.array(right_front)
            level = np.abs(right_front-left_front)

            #scaling_factor = 64.0/1000.0#For pixel to micrometer
            scaling_factor = 1.0#For pixels

            data = np.array([np.array(left_front)*scaling_factor,np.array(right_front)*scaling_factor,level*scaling_factor]).T

            data_with_positions = np.ndarray(shape=(data.shape[0],data.shape[1]+2))
            data_with_positions[:,0:-2] = data
            data_with_positions[:,-2] = self.column_y
            data_with_positions[:,-1] = self.column_x

            df = pd.DataFrame(data_with_positions, columns=["left_front","right_front","level", "pos_y", "pos_x"])
            self.measurements.emit(df)

        def set_model(self, df):
            self.dataset = df
            self.table_model = PandasModel(df)
            self.ui.polymer_level.setModel(self.table_model)

        def save_csv(self, path):
            df = self.dataset
            df.to_csv(path,";")




    class ImbibitionFront(QThread):
        """ Samples slope of grayvalues for each column """
        polymer_level_columnwise = pyqtSignal(np.ndarray)
        def __init__(self, outer):
            super(MicroTomographyAnalyzer.ImbibitionFront, self).__init__()
            self.outer = outer
            self.ui = outer.ui

            self.column_x = None
            self.column_y = None
            self.make_connections()

        def set_column_positions(self, column_positions):
            self.column_x = column_positions[0]
            self.column_y = column_positions[1]

        def make_connections(self):
            self.ui.sample_imbibition_front.clicked.connect(self.find)
            #self.polymer_level_columnwise.connect(self.display)
            self.polymer_level_columnwise.connect(self.outer.columnwise_polymer.set_polymer_level)


        def find(self):
            if type(self.column_x) == type(None) or type(self.column_y) == type(None):
                QMessageBox.information(None, "Cuould not find imbibition front","Perform column detection first.")
                return
            if type(self.outer.tensor) == type(None):
                QMessageBox.information(None, "Cuould not find imbibition front","No tensor loaded.")
                return
            self.run()#Todo change to self.start


        def run(self):
            tensor = self.outer.tensor#dimensions y, x, z, column positions are sampled for positions y and x

            w_size = 4
            through_columns = []
            pad = 2 * w_size
            padded = np.pad(tensor, pad, mode="constant", constant_values=0)
            for y, x in zip(self.column_y, self.column_x):
                column = padded[pad + int(y)-w_size:int(y)+w_size + pad,
                                pad + int(x)-w_size:int(x)+w_size + pad,
                                pad:-pad]
                column = np.nansum(np.nansum(column,axis=0),axis=0)
                through_columns.append(column)

            through_columns = np.array(through_columns)
            self.polymer_level_columnwise.emit(through_columns)
            # self.ui.table.setModel(PandasModel(df))

class Main():
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.set_color_theme(self.app, "light")

        MainWindow = QtWidgets.QMainWindow()#Create a window
        self.main_ui = Ui_MicroTomographyAnalyzer()#Instanciate our UI
        self.main_ui.setupUi(MainWindow)#Setup our UI as this MainWindow

        self.source_dir_opener = FileDialog()

        self.main_ui.centralwidget.setFocusPolicy(Qt.NoFocus)

        self.main_app = MicroTomographyAnalyzer(self.main_ui.centralwidget, self.main_ui)#Install MainApp as event filter for handling of arrow keys

        self.main_ui.centralwidget.installEventFilter(self.main_app)

        self.make_connections()
        MainWindow.show()#and we show it directly

        self.app.exec_()
        sys.exit()

    def set_color_theme(self,app, color):
        path = ""
        if color == "dark":
            path += ":/dark.qss"
            self.use_light = False
        elif color == "light":
            path += ":/light.qss"
            self.use_light = True
        else:
            return
        file = QFile(path)
        file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(file)
        app.setStyleSheet(stream.readAll())

    def make_connections(self):
        self.main_ui.actionOpen.triggered.connect(self.source_dir_opener.get_folder_path)
        self.source_dir_opener.folder.connect(self.main_app.set_source_dir)
        self.main_ui.actionSave.triggered.connect(self.save_current_tensor)
        self.main_ui.actionOpen_Tensor.triggered.connect(self.load_tensor)

    def save_current_tensor(self):
        if type(self.main_app.tensor) == type(None):
            QMessageBox.information(None, "Saving not possible","There is no tensor data")
        dialog = FileDialog(".npy")
        path = dialog.create_output_file()

        try:
            with open(path, "wb") as f:
                np.save(path, self.main_app.tensor)
        except Exception as e:
            QMessageBox.information(None, "Saving not possible","There was a problem writing the file " + str(e))

    def load_tensor(self):
        dialog = FileDialog(".npy")
        path = dialog.open_file()
        try:
            with open(path, "rb") as f:
                tensor = np.load(f)
            self.main_app.set_tensor(tensor)
        except Exception as e:
            QMessageBox.information(None, "Loading not possible","There was a problem loading the file " + str(e))

if __name__ == "__main__":
    m = Main()
