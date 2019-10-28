from view import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget, QMessageBox, QMenu
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import pyqtSignal, Qt, QModelIndex, QPoint, QRect
from PyQt5.QtGui import QStandardItemModel, QCursor
from image_operations import *
import sys
import re
import os
import shutil
from PIL import Image
from PyQt5.QtCore import QFile, QTextStream#Dark theme
import breeze_resources
import pickle
from qevent_to_name import *
from selectboxoverlay import *

import hashlib
import ast

#87, 34
class FileDialog(QWidget):
    outfilepath = pyqtSignal(str)
    folder = pyqtSignal(str)
    filepath = pyqtSignal(str)

    def __init__(self, file_ending = ""):
        self.file_ending = file_ending
        QWidget.__init__(self)

    def create_output_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(None,"Select the output file", "", self.file_ending +" (*."+self.file_ending+");;", options=options)

        if filename:
            if not filename.endswith(".pkl"):
                filename += ".pkl"
            self.outfilepath.emit(filename)

    def get_file(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter(self.file_ending +" (*."+self.file_ending+");;")
        #filenames = QStringList()
        filenames = None

        if dialog.exec_():
            filenames = dialog.selectedFiles()
            self.filepath.emit(filenames[0])

    def get_existing_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(None,"Select the output file", "", self.file_ending +" (*."+self.file_ending+");;", options=options)
        if filename:
            self.filepath.emit(filename)

    def get_folder_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(None,"Select folder...",os.getcwd(),options=options)
        if path:
            self.folder.emit(path)

class MainApp(QWidget):
    def __init__(self, widget_handled, ui):
        """ Initializes Main App.
        args:
            widget_handled: For this widget events are handeled by MainApp
            ui: User interface
        """
        self.ui = ui
        self.widget_handled = widget_handled #Widget for event Handling. All events are checked and if not arrow keys passed on. See eventFilter below
        QWidget.__init__(self, widget_handled)

        self.model_cascade_1 = self.get_cascade_model(self.ui.view_cascade_1, self.ui.combo_box_cascade_1)
        self.model_cascade_2 = self.get_cascade_model(self.ui.view_cascade_2, self.ui.combo_box_cascade_2)

        self.files = []#list of input filepaths (each filepath begins with self.files_root_directory)
        self.files_root_directory = ""#Root directory of input files
        self.tempfiles_path = os.getcwd()+"/tempfiles/"

        self.use_cropped = False

        self.cascade_outfile_dialog_1 = FileDialog("pkl")
        self.cascade_loadfile_dialog_1 = FileDialog("pkl")

        self.cascade_specify_output_1 = FileDialog()
        self.output_location = ""

        self.cascade_outfile_dialog_2 = FileDialog("pkl")
        self.cascade_loadfile_dialog_2 = FileDialog("pkl")

        self.cascade_load_attributions_1 = FileDialog("ini")
        self.cascade_load_attributions_2 = FileDialog("ini")

        self.attributions_cascade_1 = {}
        self.attributions_cascade_2 = {}

        self.filename_attributions_1 = os.getcwd()+"/"+"attributions_cascade_1.ini"#Load default attributions
        self.filename_attributions_2 = os.getcwd()+"/"+"attributions_cascade_2.ini"
        self.load_attributions(self.filename_attributions_1,self.filename_attributions_2)

        #Create selectboxoverlay &  specify variable that specifies where the cropped values should end up
        self.box = SelectBoxOverlay(self.ui.main_plot)
        self.target_of_selected_coordinates = None#Cropped values should end up here
        self.box.coordinates.connect(self.append_selection_to_parameters)

        self.use_attributions_cascade_1 = True
        self.use_attributions_cascade_2 = True

        self.idx_image = 0

        #Set context menu       #this works for left click...self.ui.view_cascade_1.clicked.connect(lambda index: self.treeMedia_doubleClicked(index,self.ui.view_cascade_1))
        self.ui.view_cascade_1.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.view_cascade_1.customContextMenuRequested.connect(lambda pos: self.show_cascade_context_menu(pos, self.ui.view_cascade_1,self.model_cascade_1))

        #Establish connections between gui and backend
        self.make_connections()

    def specify_output_location(self, folder):
        print("creating output")
        self.output_location = folder
        for f in self.files:#Create a nested folder structure that will contain one .csv file per input file
            if os.path.isdir(f):
                os.makedirs(self.output_location+"/"+f[len(self.files_root_directory):], exist_ok=True)
        #except: #TODO

    def append_selection_to_parameters(self, coordinates):
        if type(self.target_of_selected_coordinates) == type(None):
            return
        try:
            coordinates = np.array([coordinates.top(),coordinates.left(),coordinates.bottom(),coordinates.right()],dtype=np.float32)
            coordinates[0] -= self.ui.main_plot.scaled_img_margin_top#Coordinates are relative to qlabel -> Substract upper left point
            coordinates[2] -= self.ui.main_plot.scaled_img_margin_top
            coordinates[1] -= self.ui.main_plot.scaled_img_margin_left
            coordinates[3] -= self.ui.main_plot.scaled_img_margin_left
            coordinates *= self.ui.main_plot.img_scaling_ratio#Transform coordinates into pixels of the rendered plot

            coordinates[0] -= 34#Substract upper left corner in matplotlib plot where depiction begins
            coordinates[1] -= 87
            coordinates[2] -= 34#Substract upper left corner in matplotlib plot where depiction begins
            coordinates[3] -= 87
            #transform scale from matplib_pixels to original_pixels
            shape_before_plotting = np.array(Image.open(self.files[self.idx_image])).shape
            y_scaling_ratio_rendering = shape_before_plotting[0]/1495#1520
            x_scaling_ratio_rendering = shape_before_plotting[1]/2268#2230

            coordinates[0] *= y_scaling_ratio_rendering
            coordinates[1] *= x_scaling_ratio_rendering
            coordinates[2] *= y_scaling_ratio_rendering
            coordinates[3] *= x_scaling_ratio_rendering

            coordinates = list(np.array(np.round(coordinates), dtype=np.int32))


            old = ast.literal_eval(self.target_of_selected_coordinates.text())
            old.append(coordinates)
            self.target_of_selected_coordinates.setText(str(old))
            self.target_of_selected_coordinates = None
        except Exception as e:
            print(e)

    def show_cascade_context_menu(self, pos, tree_view, model):
        assert type(model) != type(None)
        x = tree_view.indexAt(pos).column()
        y = tree_view.indexAt(pos).row()
        name = model.item(y,0).text()

        if name == "mask values": # Mask values has two (string represenations) of lists as parameters.
            # The first specifies which values should be removed from the 1D array. When option in dropdown menu is selectedFiles
            # We set self.target_of_selected_coordinates accordingly. Cropping will append values to the list
            add_to_mask = model.item(y,1)#First parameter (string that represents a list)
            remove_from_mask = model.item(y,2)#Second parameter

            self.popup_menu = QMenu(self)
            self.popup_menu.addAction("Mask out values ...", lambda: self.set_target_of_selected_coordinates(add_to_mask))
            #self.popup_menu.addSeparator()
            self.popup_menu.addAction("Bring back values ...", lambda: self.set_target_of_selected_coordinates(remove_from_mask))
            self.popup_menu.popup(QCursor.pos())

    def set_target_of_selected_coordinates(self,target):
        self.target_of_selected_coordinates = target


    def load_attributions_cascade_1(self, filepath):
        self.load_attributions(attributions_cascade_1=filepath)
        self.filename_attributions_1 = filepath
        self.update_model_via_attributions(1)

    def load_attributions_cascade_2(self, filepath):
        self.load_attributions(attributions_cascade_2=filepath)
        self.filename_attributions_2 = filepath
        self.update_model_via_attributions(2)

    def load_attributions(self, attributions_cascade_1=None,attributions_cascade_2=None):
        if attributions_cascade_1:
            try:
                with open(attributions_cascade_1, "rb") as f:
                    self.attributions_cascade_1 = pickle.load(f)
            except Exception as e:
                print(e)
        if attributions_cascade_2:
            try:
                with open(attributions_cascade_2, "rb") as f:
                    self.attributions_cascade_2 = pickle.load(f)
            except Exception as e:
                print(e)

    def shut_down(self):
        try:#Cleat temporary files
            shutil.rmtree(self.tempfiles_path)
        except:
            pass

        try:
            with open(self.filename_attributions_1, "wb") as f:
                pickle.dump(self.attributions_cascade_1, f)
        except Exception as e:
            QMessageBox.about(self, "Fatal error", "Couldn't save attributions\n" + str(e))
            return

        try:
            with open(self.filename_attributions_2, "wb") as f:
                pickle.dump(self.attributions_cascade_2, f)
        except Exception as e:
            QMessageBox.about(self, "Fatal error", "Couldn't save attributions\n" + str(e))
            return
        sys.exit()

    def remember_attributions(self, cascade_no):
        hash = self.get_hash()
        if cascade_no == 1:
            if not self.use_attributions_cascade_1:
                return
            self.attributions_cascade_1[hash]=self.cascade_model_to_list(self.model_cascade_1)
        elif cascade_no == 2:
            if not self.use_attributions_cascade_2:
                return
            self.attributions_cascade_2[hash]=self.cascade_model_to_list(self.model_cascade_2)
        else:
            raise Exception("No such cascade")

    def get_hash(self):
        """ Returns hash value for current image based on it's filename and a hash of the values"""
        path = self.files[self.idx_image]
        filename = path.split("/")[-1]
        with open(path,"rb") as f:
            hash_object = hashlib.sha512(f.read())
            hex_dig = hash_object.hexdigest()
        hash = filename + " "+ hex_dig
        return hash

    def update_model_via_attributions(self, cascade_no):
        #Get correct attributions and check if work is necessary...
        attributions = None
        model = None
        if cascade_no == 1:
            attributions = self.attributions_cascade_1
            model = self.model_cascade_1
            if not self.use_attributions_cascade_1:
                return
        elif cascade_no == 2:
            attributions = self.attributions_cascade_2
            model = self.model_cascade_2
            if not self.use_attributions_cascade_2:
                return
        else:
            raise Exception("No such cascade")

        try:
            hash = self.get_hash()
        except Exception as e:
            print("Problem loading hash")
            print(e)
            return

        if hash in attributions.keys():
            cascade = attributions[hash]
            model.removeRows(0,model.rowCount())
            for row in cascade:
                self.add_to_cascade_model(model, *row)#Add all items to model
        else:
            #print(hash)
            attributions.keys()

    def load_files(self, path):
        self.files_root_directory = path
        self.files = []#list of filepaths
        self.list_files(path, ".*\.docx")
        docx = self.files
        for d in docx:
            outpath = self.tempfiles_path + (d.split("/")[-1])[:-5]+"/"
            extract_pics_from_docx(d, outpath)
        self.files = []
        self.list_files(path, ".*\.tif")
        if len(docx) >0:
            self.list_files(self.tempfiles_path, ".*\.png")
        self.show_image()

        self.update_model_via_attributions(1)
        self.update_model_via_attributions(2)

    def toggle_use_crop(self, state):
        self.use_cropped = state

    def list_files(self, path, regex):
        for f in os.scandir(path):
            if f.is_dir():
                self.list_files(f.path, regex)
            else:
                if re.match(regex,f.name):
                    self.files.append(path+"/"+f.name)

    def make_connections(self):
        """ Establishes the connections between control elements and methods that implement the respective functionalities"""
        #Connect image controls (next image, previous image etc.) ...
        self.ui.use_cropped.stateChanged.connect(self.toggle_use_crop)
        self.ui.next_image.clicked.connect(self.next_image)
        self.ui.previous_image.clicked.connect(self.previous_image)
        self.ui.show_image.clicked.connect(lambda: self.set_current_index(self.idx_image))

        #Connections for filter cascade 1...
        self.ui.add_to_cascade_1.clicked.connect(lambda: self.add_function_to_cascade(self.ui.combo_box_cascade_1,self.model_cascade_1))
        self.ui.remove_from_cascade_1.clicked.connect(lambda: self.remove_selected_from_model(self.model_cascade_1, self.ui.view_cascade_1))
        self.ui.apply_cascade_1.clicked.connect(lambda: self.apply_cascade(self.model_cascade_1,self.ui.error_cascade_1))
        self.ui.save_cascade_1.clicked.connect(self.cascade_outfile_dialog_1.create_output_file)
        self.cascade_outfile_dialog_1.outfilepath.connect(lambda path: self.save_cascade_model(self.model_cascade_1,path))
        self.ui.load_cascade_1.clicked.connect(self.cascade_loadfile_dialog_1.get_existing_file)
        self.cascade_loadfile_dialog_1.filepath.connect(lambda path: self.load_cascade_model(self.model_cascade_1,path))

        #Connections for filter cascade 1: Connectiond for loading attribution files (that store the cacades for all images)...
        self.ui.load_attributions_cascade_1.clicked.connect(self.cascade_load_attributions_1.get_file)
        self.cascade_load_attributions_1.filepath.connect(self.load_attributions_cascade_1)

        self.ui.specify_output_cascade_1.clicked.connect(self.cascade_specify_output_1.get_folder_path)
        self.cascade_specify_output_1.folder.connect(self.specify_output_location)

        #Connections for filter cascade 2...
        self.ui.add_to_cascade_2.clicked.connect(lambda: self.add_function_to_cascade(self.ui.combo_box_cascade_2,self.model_cascade_2))
        self.ui.remove_from_cascade_2.clicked.connect(lambda: self.remove_selected_from_model(self.model_cascade_2, self.ui.view_cascade_2))
        self.ui.apply_cascade_2.clicked.connect(lambda: self.apply_cascade(self.model_cascade_2,self.ui.error_cascade_2))
        self.ui.save_cascade_2.clicked.connect(self.cascade_outfile_dialog_1.create_output_file)
        self.cascade_outfile_dialog_2.outfilepath.connect(lambda path: self.save_cascade_model(self.model_cascade_2,path))
        self.ui.load_cascade_2.clicked.connect(self.cascade_loadfile_dialog_2.get_existing_file)
        self.cascade_loadfile_dialog_2.filepath.connect(lambda path: self.load_cascade_model(self.model_cascade_2,path))

        #Connections for filter cascade 1: Connectiond for loading attribution files (that store the cacades for all images)...
        self.ui.load_attributions_cascade_2.clicked.connect(self.cascade_load_attributions_2.get_file)
        self.cascade_load_attributions_2.filepath.connect(self.load_attributions_cascade_2)


    def apply_cascade(self, model, error_view):
        """ Applies cascade and saves results if self.output_location was set"""
        img = None
        try:
            img = Image.open(self.files[self.idx_image])
            img = np.array(img.convert('I'), dtype=np.double)#Convert to grayscale
        except:
            return

        self.box.toggle_enabled()#Avoid danger of segfault

        operations = self.cascade_model_to_list(model)
        data = image_to_cascade_data(img)
        try:
            error_view.setText("")
            for o in operations:
                data = name_to_function[o[0]](data,*o[1:])

            plot = None
            slope_intercept = None
            background_image = img
            if "houghlines" in data and self.ui.draw_vertica_lines.isChecked():
                slope_intercept = data["houghlines"]
            if "image_array" in data:
                background_image = data["image_array"]
            if "1d_array" in data:
                plot = data["1d_array"]
            if self.use_cropped and "cropped_original" in data:
                background_image = data["cropped_original"]
            if "columnwise" in data:
                measurements = ""
                measurements += "Column number \t\t|\t Polymer level\n"
                for i, x in enumerate(data["columnwise"]):
                    measurements += "Column " + str(i) + "\t\t|\t" + str(x) + " μm\n"
                error_view.setText(measurements)


            self.ui.main_plot.update(color_plot(background_image, plot = plot, slope_intercept = slope_intercept))

            if self.output_location != "":
                self.save_output_for_current_file(data["columnwise"])

        except Exception as e:
            error_view.setText(str(e))
        finally:
            self.box.toggle_enabled()

    def save_output_for_current_file(self,data):
        current_file = self.files[self.idx_image]
        subpath = current_file[len(self.files_root_directory):]#path in nested directory without root
        outfile = self.output_location+subpath.split(".")[0]+".txt"#replace file_ending
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        measurements = ""
        measurements += "Column number \t\t|\t Polymer level\n"
        for i, x in enumerate(data):
            measurements += "Column " + str(i) + "\t\t|\t" + str(x) + " μm\n"
        with open(outfile,"w") as f:
            f.write(measurements)


    def cascade_model_to_list(self, model):
        data = []
        for y in range(model.rowCount()):
            row = []
            for x in range(model.columnCount()):
                item = model.item(y,x)
                if item and item.text() != "":
                    try:
                        row.append(int(item.text()))
                    except:
                        try:
                            row.append(float(item.text()))
                        except:
                            row.append(item.text())
            data.append(row)
        return data

    def save_cascade_model(self, model, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.cascade_model_to_list(model),f)

    def load_cascade_model(self, model, filename):
        with open(filename, "rb") as f:
            cascade = pickle.load(f)
            for y in range(model.rowCount()):
                model.removeRows(y,1)
            for row in cascade:
                self.add_to_cascade_model(model, *row)

    def add_function_to_cascade(self, combo_box, model):
        function = combo_box.currentText()
        parameters = function_to_default_params[function]
        parameters = [str(e) for e in parameters]
        parameters.insert(0,function)
        self.add_to_cascade_model(model,*parameters)

    def next_image(self):
        self.set_current_index(self.idx_image+1)

    def set_current_index(self, idx):
        if idx < 0 or idx >= len(self.files):
            print("no such index")
            return
        self.remember_attributions(1)
        self.remember_attributions(2)
        self.idx_image = idx
        self.ui.file_number.setValue(idx)
        self.show_image()
        self.update_model_via_attributions(1)
        self.update_model_via_attributions(2)

    def previous_image(self):
        self.set_current_index(self.idx_image-1)

    def show_image(self):
        try:
            path =  self.files[self.idx_image]
            img = Image.open(path)
        except:
            return
        img = img.convert('RGBA')

        name =  path.split("/")[-1] #set filename
        if "tempfiles" in path:
            name =  path.split("/")[-2] + "/" + path.split("/")[-1]
        self.ui.filenames.setText(path)

        if str(img.mode) != "RGBA":
            img = img.convert('RGB')
        img = color_plot(np.array(img.convert('I'), dtype=np.double))#np.array(img,dtype=np.byte)
        self.ui.main_plot.update(img)

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
                self.next_image()

            elif event.key() == id_left:
                self.previous_image()
        try:#When closing the app the widget handled might already have been destroyed
            return self.widget_handled.eventFilter(source, event)#Execute the default actions for the event
        except:
            return True#a true value prevents the event from being sent on to other objects

    def get_cascade_model(self, view, combobox):
        model = QStandardItemModel(0, 5, self)
        model.setHeaderData(0, Qt.Horizontal, "Function name")
        for i in range(1,5):
            model.setHeaderData(i, Qt.Horizontal, "param " + str(i))
        view.setModel(model)

        for function in name_to_function:
            combobox.addItem(function)
        return model

    def add_to_cascade_model(self, model, *strings):
        current_row = model.rowCount()
        model.insertRow(current_row)#create QStandardItem(..) and use appendRow instead?

        root = model.invisibleRootItem()#To set flags
        for i, s in enumerate(strings):
            model.setData(model.index(current_row, i), str(s))
            item = root.child(current_row,i)
            item.setFlags( Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled | Qt.ItemIsEditable)#Allow only top level drop

    def remove_selected_from_model(self, model, view):
        idxs = view.selectedIndexes()
        if idxs and len(idxs) > 0:
            model.removeRows(idxs[0].row(),1)


class Kapilar():
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.set_color_theme(self.app, "light")

        MainWindow = QtWidgets.QMainWindow()#Create a window
        self.ui = Ui_kapilar()#Instanciate our UI
        self.ui.setupUi(MainWindow)#Setup our UI as this MainWindow

        self.source_dir_opener = FileDialog()

        self.main_app = MainApp(self.ui.centralwidget, self.ui)#Install MainApp as event filter for handling of arrow keys

        self.ui.centralwidget.installEventFilter(self.main_app)

        self.make_connections()
        MainWindow.show()#and we show it directly

        self.app.exec_()
        self.main_app.shut_down()
        #sys.exit()

    def toggle_color_scheme(self):
        app = QApplication.instance()
        if app is None:
            raise RuntimeError("No Qt Application found.")
        if self.use_light:
            self.set_color_theme(app,"dark")
        else:
            self.set_color_theme(app,"light")


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
        self.ui.open_file.triggered.connect(self.source_dir_opener.get_folder_path)
        self.ui.exit.triggered.connect(self.main_app.shut_down)
        self.source_dir_opener.folder.connect(self.main_app.load_files)
        self.ui.toggle_colors.triggered.connect(self.toggle_color_scheme)

if __name__ == "__main__":
    k = Kapilar()
