from view import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import pyqtSignal, Qt, QModelIndex
from PyQt5.QtGui import QStandardItemModel
from image_operations import *
import sys
import re
import os
import shutil
from PIL import Image
from PyQt5.QtCore import QFile, QTextStream#Dark theme
import breeze_resources

from qevent_to_name import *

class SourceDirOpener(QWidget):
    folder = pyqtSignal(str)
    def __init__(self):
        QWidget.__init__(self)
    def get_filenames(self):
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

        self.model_2d_cascade = self.get_cascade_model()
        assert type(self.model_2d_cascade) != type(None)
        self.files = []#list of filepaths
        self.tempfiles_path = os.getcwd()+"/tempfiles/"

        self.idx_image = 0
        self.make_connections()


    def shut_down(self):
        shutil.rmtree(self.tempfiles_path)
        sys.exit()

    def load_files(self, path):
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

    def list_files(self, path, regex):
        for f in os.scandir(path):
            if f.is_dir():
                self.list_files(f.path, regex)
            else:
                if re.match(regex,f.name):
                    self.files.append(path+"/"+f.name)

    def make_connections(self):
        self.ui.next_image.clicked.connect(self.next_image)
        self.ui.previous_image.clicked.connect(self.previous_image)
        self.ui.add_to_2d_cascade.clicked.connect(lambda: self.add_function_to_cascade(self.ui.combo_box_2d_cascade,self.model_2d_cascade))
        self.ui.remove_from_2d_cascade.clicked.connect(lambda: self.remove_selected_from_model(self.model_2d_cascade, self.ui.view_2d_cascade))
        self.ui.apply_cascade_2d.clicked.connect(self.apply_cascade)

    def apply_cascade(self):
        try:
            img = Image.open(self.files[self.idx_image])
            img = img.convert('RGB')
        except:
            #TODO: return
            pass
        operations = self.cascade_model_to_list( self.model_2d_cascade)
        try:
            pass
        except:
            pass

    def cascade_model_to_list(self, model):
        for y in range(model.rowCount()):
            for x in range(model.columnCount()):
                item = model.item(y,x)
                if item:
                    print(item.text())


    def add_function_to_cascade(self, combo_box, model):
        function = combo_box.currentText()
        parameters = function_to_default_params[function]
        parameters = [str(e) for e in parameters]
        parameters.insert(0,function)
        self.add_to_cascade_model(model,*parameters)

    def next_image(self):
        if self.idx_image+1 == len(self.files):
            return
        self.idx_image += 1
        self.show_image()

    def previous_image(self):
        if self.idx_image == 0:
            return
        self.idx_image -= 1
        self.show_image()

    def show_image(self):
        img = Image.open(self.files[self.idx_image])
        if str(img.mode) != "RGBA":
            img = img.convert('RGB')
        img = np.array(img,dtype=np.byte)
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
        return self.widget_handled.eventFilter(source, event)#forward event

    def get_cascade_model(self):
        model = QStandardItemModel(0, 5, self)
        model.setHeaderData(0, Qt.Horizontal, "Function name")
        for i in range(1,5):
            model.setHeaderData(i, Qt.Horizontal, "param " + str(i))
        self.ui.view_2d_cascade.setModel(model)

        for function in name_to_function:
            self.ui.combo_box_2d_cascade.addItem(function)
        return model

    def add_to_cascade_model(self, model, *strings):
        current_row = model.rowCount()
        model.insertRow(current_row)#create QStandardItem(..) and use appendRow instead?

        root = model.invisibleRootItem()#To set flags
        for i, s in enumerate(strings):
            model.setData(model.index(current_row, i), s)
            item = root.child(current_row,i)
            item.setFlags( Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled | Qt.ItemIsEditable)#Allow only top level drop


    def remove_selected_from_model(self, model, view):
        idxs = view.selectedIndexes()
        if idxs and len(idxs) > 0:
            model.removeRows(idxs[0].row(),1)




class Kapilar():
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        self.set_color_theme(app, "light")

        MainWindow = QtWidgets.QMainWindow()#Create a window
        self.ui = Ui_kapilar()#Instanciate our UI
        self.ui.setupUi(MainWindow)#Setup our UI as this MainWindow

        self.source_dir_opener = SourceDirOpener()

        self.main_app = MainApp(self.ui.centralwidget, self.ui)#Install MainApp as event filter for handling of arrow keys

        self.ui.centralwidget.installEventFilter(self.main_app)

        self.make_connections()
        MainWindow.show()#and we show it directly

        app.exec_()
        self.main_app.shut_down()
        sys.exit()

    def set_color_theme(self,app, color):
        path = ""
        if color == "dark":
            path += ":/dark.qss"
        elif color == "light":
            path += ":/light.qss"
        else:
            return
        file = QFile(path)
        file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(file)
        app.setStyleSheet(stream.readAll())

    def make_connections(self):
        self.ui.open_file.triggered.connect(self.source_dir_opener.get_filenames)
        self.ui.exit.triggered.connect(self.main_app.shut_down)
        self.source_dir_opener.folder.connect(self.main_app.load_files)

if __name__ == "__main__":
    k = Kapilar()
