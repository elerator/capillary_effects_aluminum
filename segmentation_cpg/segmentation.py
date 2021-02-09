from segmentation_ui import Ui_Segmentation
import breeze_resources#light layout
from PyQt5 import QtCore, QtWidgets, QtWebChannel
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QFile, QTextStream

import sys
import os
import traceback

import numpy as np
import pandas as pd
import PIL
from PIL import Image

import json
from skimage import exposure

from imagedisplay import fig2rgb_array
from plotly_voxel_plot import *

class FileDialog(QWidget):
    outfilepath = pyqtSignal(str)
    folder = pyqtSignal(str)
    #filepath = pyqtSignal(str)

    def __init__(self, file_ending = ".csv"):
        """ A File dialog could either be used to show a dialog to create and output file (i.e. to get a nonexisting path),
            to open a file or to get the name of a folder. The respective member functions may be used in this respect. The filepath is returned and emitted as a pyqtSignal
        Args:
            file_ending: The file ending of the file to be selected (Use empty string for Folder selection)
        """
        self.file_ending = file_ending
        self.columns = None
        QWidget.__init__(self)

    def create_output_file(self):
        """ Opens dialog for non-existing files and returns path.
        Returns:
            Path to a non-existing file (str). If the specified filename does not end with self.filepath the respective ending is added.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(None,"Select the output file", "", self.file_ending[:] +" (*."+self.file_ending[1:]+");;", options=options)

        if filename:
            if not filename.endswith(self.file_ending):
                filename += self.file_ending
            self.outfilepath.emit(filename)
        return filename

    def open_file(self):
        """ Opens a file dialog for existing files. Emits path as signal outfilepath.
        Returns:
            Path to existing file. The ending specified in self.filepath is added if the file does not already end with named string (str)
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(None,"Select the output file", "", self.file_ending[1:] +" (*."+self.file_ending[1:]+");;", options=options)
        self.outfilepath.emit(filename)
        return filename


    def get_folder_path(self):
        """ Opens dialog for folder selection. Emits path as signal folder.
        Returns:
            Path to folder (str)
        """
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

        self.Mesh3d = MicroTomographyAnalyzer.Mesh3dDisplay(self)

        self.slicer = MicroTomographyAnalyzer.Slicer(self)
        self.rotator = MicroTomographyAnalyzer.Rotator(self)

        self.slice_xy = True
        self.slice_xz = False
        self.slice_yz = False
        self.current_slice = 0

        #Prepare the webview
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"react/plot_3d/build/index.html")
        self.ui.webEngineView.setUrl(QtCore.QUrl.fromLocalFile(file))
        channel = self.channel = QtWebChannel.QWebChannel()
        channel.registerObject("MainWindow", self)
        self.ui.webEngineView.page().setWebChannel(channel)
        self.make_connections()

        #self.ui.update_plot.clicked.connect(lambda: self.set_mesh(random.sample(range(10, 30), 20),random.sample(range(10, 30), 20),random.sample(range(10, 30), 20)))


    def set_display_axis(self, axis):
        """ Sets axis that is used to slice tensor for display purposes """
        if axis == "yx":
            self.slice_xy = True
            self.slice_xz = False
            self.slice_yz = False
        elif axis == "xz":
            self.slice_xy = False
            self.slice_xz = True
            self.slice_yz = False
        elif axis == "yz":
            self.slice_xy = False
            self.slice_xz = False
            self.slice_yz = True
        self.display_slice(0)

    def make_connections(self):
        """ Esdtablishes connections between GUI and fucntionalities."""
        self.tensor_loader.tensor.connect(self.set_tensor)
        self.tensor_loader.update_info.connect(self.ui.progress.setValue)
        self.ui.next.clicked.connect(lambda: self.display_slice(self.current_slice+1))
        self.ui.previous.clicked.connect(lambda: self.display_slice(self.current_slice-1))
        self.ui.axis.currentTextChanged.connect(self.set_display_axis)
        self.slicer.output_tensor.connect(self.set_tensor)
        self.ui.actionRotate.triggered.connect(lambda x: self.ui.tools.setCurrentWidget(self.ui.page_rotating))
        self.ui.actionRotate.triggered.connect(lambda x: self.ui.display_stack.setCurrentWidget(self.ui.page_slices))
        self.ui.actionSlice_tensor.triggered.connect(lambda x: self.ui.tools.setCurrentWidget(self.ui.page_slicing))
        self.ui.actionSlice_tensor.triggered.connect(lambda x: self.ui.display_stack.setCurrentWidget(self.ui.page_slices))
        self.ui.action3D_visualization.triggered.connect(lambda: self.ui.tools.setCurrentWidget(self.ui.page_3d_generator))
        self.ui.action3D_visualization.triggered.connect(lambda: self.ui.display_stack.setCurrentWidget(self.ui.page_3d_plot))

        self.ui.next_display_page.clicked.connect(self.next_display_page)
        self.ui.previous_display_page.clicked.connect(self.previous_display_page)


    def next_display_page(self):
        """ Display next page (Slices, Detected pores or measured polymer level)"""
        if self.ui.display_stack.currentIndex()+1 == self.ui.display_stack.count():
            return
        try:
            self.ui.display_stack.setCurrentIndex(self.ui.display_stack.currentIndex()+1)
        except:
            pass

    def previous_display_page(self):
        """ Display previous page (Slices, Detected pores or measured polymer level)"""
        if self.ui.display_stack.currentIndex() == 0:
            return
        try:
            self.ui.display_stack.setCurrentIndex(self.ui.display_stack.currentIndex()-1)
        except:
            pass

    def set_source_dir(self, source_dir):
        """ Sets path to source directory """

        self.source_dir = source_dir
        files = [os.path.join(self.source_dir,x) for x in os.listdir(self.source_dir)]
        self.ui.loading_info_stack.setCurrentIndex(1)
        self.tensor_loader.set_files(files)
        self.tensor_loader.start()

    def set_tensor(self, tensor):
        """ Sets 3D data tensor
        Args:
            tensor: 3D Numpy array
        """
        self.tensor = tensor
        self.ui.loading_info_stack.setCurrentIndex(0)
        self.ui.file_info.setText("Loaded tensor successfully")
        self.display_slice(self.current_slice)
        self.slicer.update_axis_bounds()

    def get_tensor(self):
        """ Returns current tensor """
        return self.tensor

    def get_current_slice(self):
        """ Getter for current slice"""
        slice = None
        idx = self.current_slice
        if self.slice_xy:
            slice = self.tensor[:,:,idx].copy()#remove copy in this line?
        elif self.slice_yz:
            slice = self.tensor[:,idx,:].copy()
        elif self.slice_xz:
            slice = self.tensor[idx,:,:].copy()#TODO: Check if correct...
        return slice

    def normalize(self, tensor):
        tensor = tensor - np.min(tensor)
        tensor = tensor / np.max(tensor)
        return tensor

    def improve_slice(self, slice):
        """ Improve slice before displaying """
        if self.ui.adapthist_preview.isChecked():
            slice = exposure.equalize_adapthist(self.normalize(slice),int(self.ui.adapthist_preview_kernelsize.value()), clip_limit=0.03)
        if self.ui.do_threshold_preview.isChecked():
            slice = self.normalize(slice)
            thres = self.ui.threshold_preview.value()
            slice = slice > thres
            slice = slice.astype(np.float)
        return slice

    def display_slice(self, idx):
        """ Displayes current slicev"""
        old_idx = self.current_slice
        try:
            self.current_slice = idx
            self.ui.slices.update(self.improve_slice(self.get_current_slice()))
            self.ui.slice.setValue(self.current_slice)
        except Exception as err:
            self.current_slice = old_idx
            print(err)
            traceback.print_tb(err.__traceback__)

    def set_column_positions(self, columns):
        """ Sets value for indicator that shows column potistions """
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
            """ Thread for loading tensor from image files in parallel"""
            super(MicroTomographyAnalyzer.TensorLoader, self).__init__()
            self.files = None

        def set_files(self,files):
            """ Sets filenames"""
            self.files = files

        def run(self):
            """ Loads tensor. Emits data as self.tensor (PyQtSignal)."""
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


    class Mesh3dDisplay(QThread):
        progress = pyqtSignal(int)
        data = pyqtSignal(str)
        def __init__(self,outer):
            super(MicroTomographyAnalyzer.Mesh3dDisplay,self).__init__()
            self.ui = outer.ui
            self.outer = outer
            self.json = None
            self.make_connections()

        def make_connections(self):
            """ Establishes connections between GUI elements and methods"""
            self.progress.connect(self.ui.progress_mesh.setValue)
            self.ui.generate_mesh.clicked.connect(lambda: self.start() if type(self.outer.tensor)!=type(None) else None)
            self.data.connect(self.set_mesh)
            self.data.connect(self.set_json)

        def set_json(self, data):
            self.json = data

        def save_json(self):
            f = FileDialog()
            file = f.create_output_file()
            if not file:
                return
            try:
                with open(file, "w") as f:
                    if self.json:
                        f.write(self.json)
                    else:
                        raise Exception("")
            except:
                response = QMessageBox.information(None, "Saving not possible","A problem occured")


        def load_json(self):
            f = FileDialog()
            file = f.open_file()
            if file:
                try:
                    with open(file, "r") as f:
                        self.json = f.read()
                except:
                    response = QMessageBox.information(None, "Loading not possible","A problem occured")


        def normalize(self, tensor):
            tensor = tensor - np.min(tensor)
            tensor = tensor / np.max(tensor)
            return tensor

        def set_mesh(self, data, graph_id="main"):
            #string = json.dumps({"x":x,"y":y,"z":z})

            data = "'" + data + "'"
            self.ui.webEngineView.page().runJavaScript("Graph"+"_"+graph_id+".set_mesh("+data+")");

        def run(self):
            """ Working method for parallel rotation of whole 3d tensor"""
            tensor = self.outer.tensor.copy()
            n_slices = len(tensor)
            for i, slice in enumerate(tensor):
                slice = exposure.equalize_adapthist(self.normalize(slice),int(self.ui.adapthist_preview_kernelsize.value()), clip_limit=0.03)
                tensor[i] = slice
                self.progress.emit(int((i/n_slices)*100))
            self.progress.emit(0)
            tensor = self.normalize(tensor)
            thres = self.ui.threshold_preview.value()
            tensor = tensor > thres

            mesh = voxels_to_mesh(tensor, opacity=1.0, format="json", progress_callback= lambda x: self.progress.emit(int(x*100)))
            np.save("tensor", tensor)
            self.data.emit(mesh)

    class Slicer(QWidget):
        output_tensor = pyqtSignal(np.ndarray)
        def __init__(self, parent_controller):
            """ Slices tensor according to parameters set via GUI"""
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
            """ Sets text for axis bounds to respective to the respective labels in the GUI"""
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
            #slice = self.parent_controller.get_current_slice()
            #shape = slice.shape

            print(shape)
            low_dim1.setMaximum(shape[0])
            low_dim2.setMaximum(shape[1])

            high_dim1.setMaximum(shape[0])
            high_dim2.setMaximum(shape[1])
            high_dim1.setValue(shape[0])
            high_dim2.setValue(shape[1])

        def reset_preview(self):
            """ Resets the preview showing snippet"""
            self.update_axis_bounds()
            self.slice_preview()

        def slice_preview(self):
            """ Shows slice of tensor. Slicing is achieved using values specified in UI """
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
            """ Establish conncetions between GUI and methods"""
            self.parent_controller.ui.axis.currentTextChanged.connect(lambda x: self.update_axis_bounds())

            self.parent_controller.ui.low_bound_dim1.valueChanged.connect(lambda x: self.slice_preview())
            self.parent_controller.ui.low_bound_dim2.valueChanged.connect(lambda x: self.slice_preview())
            self.parent_controller.ui.high_bound_dim1.valueChanged.connect(lambda x: self.slice_preview())
            self.parent_controller.ui.high_bound_dim2.valueChanged.connect(lambda x: self.slice_preview())
            self.parent_controller.ui.reset_slicing.clicked.connect(lambda x: self.reset_preview())
            self.parent_controller.ui.apply_slicing.clicked.connect(lambda x: self.apply_slicing())

        def apply_slicing(self):
            """ Applys slicing to the whole tensor """
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
            """ Rotates tensor around center point in two axis
            Args:
                outer: Outer instance that owns the GUI.
            """
            super(MicroTomographyAnalyzer.Rotator,self).__init__()
            self.ui = outer.ui
            self.outer = outer
            self.angle = None
            self.make_connections()

        def make_connections(self):
            """ Establishes connections between GUI elements and methods"""
            self.ui.rotation_angle.valueChanged.connect(self.rotate_preview)
            self.output_tensor.connect(self.outer.set_tensor)
            self.progress.connect(self.ui.progress.setValue)
            self.done.connect(lambda: self.ui.loading_info_stack.setCurrentIndex(0))
            self.ui.apply_rotation.clicked.connect(self.rotate_tensor)

        def rotate_preview(self, angle):
            """ Computes preview for 2d slice and displays it """
            self.angle = angle
            slice = self.outer.get_current_slice()
            slice = Image.fromarray(np.array(slice,dtype=np.uint32)).rotate(angle,resample=PIL.Image.BICUBIC)
            self.ui.slices.update(np.array(slice))

        def rotate_tensor(self):
            """ Interface method for rotating whole tensor """
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
            """ Working method for parallel rotation of whole 3d tensor"""
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


class Main(QtWidgets.QWidget):
    def __init__(self):
        """ Initializes program. Starts app, creates window and implements functions accessible via action bar."""
        self.app = QtWidgets.QApplication(sys.argv)
        self.set_color_theme(self.app, "light")

        MainWindow = QtWidgets.QMainWindow()#Create a window
        self.main_ui = Ui_Segmentation()#Instanciate our UI
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
        """ Set ui color scheme to either dark or bright
        Args:
            app: PyQt App the color scheme is applied to
            color: String specifying the color scheme. Either "dark" or "bright".

        """
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
        """ Establishes connections between actions and GUI elements"""
        self.main_ui.actionOpen.triggered.connect(self.source_dir_opener.get_folder_path)
        self.source_dir_opener.folder.connect(self.main_app.set_source_dir)
        self.main_ui.actionSave.triggered.connect(lambda: self.save_current_tensor())
        self.main_ui.actionOpen_Tensor.triggered.connect(lambda: self.load_tensor())


    def save_current_tensor(self):
        """ Saves current tensor as numpy container"""
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
        """ Loads previously saved tensor as numpy container"""
        dialog = FileDialog(".npy")
        path = dialog.open_file()
        try:
            with open(path, "rb") as f:
                tensor = np.load(f)
            self.main_app.set_tensor(tensor)
        except Exception as e:
            QMessageBox.information(None, "Loading not possible","There was a problem loading the file " + str(e))

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons
    m = Main()#start app
