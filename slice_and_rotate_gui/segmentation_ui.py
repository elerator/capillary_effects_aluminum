# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'segmentation.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Segmentation(object):
    def setupUi(self, Segmentation):
        Segmentation.setObjectName("Segmentation")
        Segmentation.resize(1749, 1611)
        self.centralwidget = QtWidgets.QWidget(Segmentation)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_13.addWidget(self.line_2)
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.widget_6 = QtWidgets.QWidget(self.splitter)
        self.widget_6.setObjectName("widget_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_6)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.display_stack = QtWidgets.QStackedWidget(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.display_stack.sizePolicy().hasHeightForWidth())
        self.display_stack.setSizePolicy(sizePolicy)
        self.display_stack.setObjectName("display_stack")
        self.page_slices = QtWidgets.QWidget()
        self.page_slices.setObjectName("page_slices")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.page_slices)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.slices = ImageDisplay(self.page_slices)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slices.sizePolicy().hasHeightForWidth())
        self.slices.setSizePolicy(sizePolicy)
        self.slices.setMinimumSize(QtCore.QSize(400, 0))
        self.slices.setText("")
        self.slices.setObjectName("slices")
        self.verticalLayout_3.addWidget(self.slices)
        self.display_stack.addWidget(self.page_slices)
        self.page_detected_columns = QtWidgets.QWidget()
        self.page_detected_columns.setObjectName("page_detected_columns")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.page_detected_columns)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.detected_columns = ImageDisplay(self.page_detected_columns)
        self.detected_columns.setText("")
        self.detected_columns.setObjectName("detected_columns")
        self.verticalLayout_12.addWidget(self.detected_columns)
        self.display_stack.addWidget(self.page_detected_columns)
        self.page_front_display = QtWidgets.QWidget()
        self.page_front_display.setObjectName("page_front_display")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.page_front_display)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.imbibition_front = ImageDisplay(self.page_front_display)
        self.imbibition_front.setText("")
        self.imbibition_front.setObjectName("imbibition_front")
        self.verticalLayout_15.addWidget(self.imbibition_front)
        self.display_stack.addWidget(self.page_front_display)
        self.verticalLayout_5.addWidget(self.display_stack)
        self.page_controls = QtWidgets.QWidget(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.page_controls.sizePolicy().hasHeightForWidth())
        self.page_controls.setSizePolicy(sizePolicy)
        self.page_controls.setObjectName("page_controls")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.page_controls)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem)
        self.previous_display_page = QtWidgets.QPushButton(self.page_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.previous_display_page.sizePolicy().hasHeightForWidth())
        self.previous_display_page.setSizePolicy(sizePolicy)
        self.previous_display_page.setMaximumSize(QtCore.QSize(25, 16777215))
        self.previous_display_page.setObjectName("previous_display_page")
        self.horizontalLayout_7.addWidget(self.previous_display_page)
        self.next_display_page = QtWidgets.QPushButton(self.page_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.next_display_page.sizePolicy().hasHeightForWidth())
        self.next_display_page.setSizePolicy(sizePolicy)
        self.next_display_page.setMaximumSize(QtCore.QSize(25, 16777215))
        self.next_display_page.setObjectName("next_display_page")
        self.horizontalLayout_7.addWidget(self.next_display_page)
        self.verticalLayout_5.addWidget(self.page_controls)
        self.widget_5 = QtWidgets.QWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_5.sizePolicy().hasHeightForWidth())
        self.widget_5.setSizePolicy(sizePolicy)
        self.widget_5.setObjectName("widget_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_5)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tools = QtWidgets.QStackedWidget(self.widget_5)
        self.tools.setObjectName("tools")
        self.page_slicing = QtWidgets.QWidget()
        self.page_slicing.setObjectName("page_slicing")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.page_slicing)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.group_slicing = QtWidgets.QGroupBox(self.page_slicing)
        self.group_slicing.setObjectName("group_slicing")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.group_slicing)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_2 = QtWidgets.QLabel(self.group_slicing)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_6.addWidget(self.label_2)
        self.widget_2 = QtWidgets.QWidget(self.group_slicing)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.low_bound_dim1 = QtWidgets.QSpinBox(self.widget_2)
        self.low_bound_dim1.setMaximum(999999999)
        self.low_bound_dim1.setObjectName("low_bound_dim1")
        self.horizontalLayout_4.addWidget(self.low_bound_dim1)
        self.high_bound_dim1 = QtWidgets.QSpinBox(self.widget_2)
        self.high_bound_dim1.setMaximum(999999999)
        self.high_bound_dim1.setObjectName("high_bound_dim1")
        self.horizontalLayout_4.addWidget(self.high_bound_dim1)
        self.verticalLayout_6.addWidget(self.widget_2)
        self.label_3 = QtWidgets.QLabel(self.group_slicing)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_6.addWidget(self.label_3)
        self.widget_3 = QtWidgets.QWidget(self.group_slicing)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.low_bound_dim2 = QtWidgets.QSpinBox(self.widget_3)
        self.low_bound_dim2.setMaximum(999999999)
        self.low_bound_dim2.setObjectName("low_bound_dim2")
        self.horizontalLayout_5.addWidget(self.low_bound_dim2)
        self.high_bound_dim2 = QtWidgets.QSpinBox(self.widget_3)
        self.high_bound_dim2.setMaximum(999999999)
        self.high_bound_dim2.setObjectName("high_bound_dim2")
        self.horizontalLayout_5.addWidget(self.high_bound_dim2)
        self.verticalLayout_6.addWidget(self.widget_3)
        self.reset_slicing = QtWidgets.QPushButton(self.group_slicing)
        self.reset_slicing.setObjectName("reset_slicing")
        self.verticalLayout_6.addWidget(self.reset_slicing)
        self.apply_slicing = QtWidgets.QPushButton(self.group_slicing)
        self.apply_slicing.setObjectName("apply_slicing")
        self.verticalLayout_6.addWidget(self.apply_slicing)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem1)
        self.verticalLayout_9.addWidget(self.group_slicing)
        self.tools.addWidget(self.page_slicing)
        self.page_rotating = QtWidgets.QWidget()
        self.page_rotating.setObjectName("page_rotating")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.page_rotating)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.groupBox_2 = QtWidgets.QGroupBox(self.page_rotating)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.rotation_angle = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.rotation_angle.setMinimum(-360.0)
        self.rotation_angle.setMaximum(360.0)
        self.rotation_angle.setObjectName("rotation_angle")
        self.verticalLayout_10.addWidget(self.rotation_angle)
        self.preview_rotation = QtWidgets.QCheckBox(self.groupBox_2)
        self.preview_rotation.setObjectName("preview_rotation")
        self.verticalLayout_10.addWidget(self.preview_rotation)
        self.apply_rotation = QtWidgets.QPushButton(self.groupBox_2)
        self.apply_rotation.setObjectName("apply_rotation")
        self.verticalLayout_10.addWidget(self.apply_rotation)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_10.addItem(spacerItem2)
        self.verticalLayout_8.addWidget(self.groupBox_2)
        self.tools.addWidget(self.page_rotating)
        self.verticalLayout_2.addWidget(self.tools)
        self.groupBox_3 = QtWidgets.QGroupBox(self.widget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.axis = QtWidgets.QComboBox(self.groupBox_3)
        self.axis.setObjectName("axis")
        self.axis.addItem("")
        self.axis.addItem("")
        self.axis.addItem("")
        self.horizontalLayout_3.addWidget(self.axis)
        self.verticalLayout_2.addWidget(self.groupBox_3)
        self.verticalLayout_13.addWidget(self.splitter)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_13.addWidget(self.line)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.loading_info_stack = QtWidgets.QStackedWidget(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loading_info_stack.sizePolicy().hasHeightForWidth())
        self.loading_info_stack.setSizePolicy(sizePolicy)
        self.loading_info_stack.setMaximumSize(QtCore.QSize(180, 16777215))
        self.loading_info_stack.setObjectName("loading_info_stack")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.page_3)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.file_info = QtWidgets.QLabel(self.page_3)
        self.file_info.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.file_info.setObjectName("file_info")
        self.horizontalLayout_2.addWidget(self.file_info)
        self.loading_info_stack.addWidget(self.page_3)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.page_4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.progress = QtWidgets.QProgressBar(self.page_4)
        self.progress.setMaximumSize(QtCore.QSize(180, 16777215))
        self.progress.setProperty("value", 0)
        self.progress.setObjectName("progress")
        self.verticalLayout.addWidget(self.progress)
        self.loading_info_stack.addWidget(self.page_4)
        self.horizontalLayout.addWidget(self.loading_info_stack)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.previous = QtWidgets.QPushButton(self.widget)
        self.previous.setMinimumSize(QtCore.QSize(80, 0))
        self.previous.setMaximumSize(QtCore.QSize(80, 16777215))
        self.previous.setObjectName("previous")
        self.horizontalLayout.addWidget(self.previous)
        self.next = QtWidgets.QPushButton(self.widget)
        self.next.setMinimumSize(QtCore.QSize(80, 0))
        self.next.setMaximumSize(QtCore.QSize(80, 16777215))
        self.next.setObjectName("next")
        self.horizontalLayout.addWidget(self.next)
        self.slice = QtWidgets.QSpinBox(self.widget)
        self.slice.setMaximum(999999999)
        self.slice.setObjectName("slice")
        self.horizontalLayout.addWidget(self.slice)
        self.verticalLayout_13.addWidget(self.widget)
        Segmentation.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Segmentation)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1749, 17))
        self.menubar.setObjectName("menubar")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        Segmentation.setMenuBar(self.menubar)
        self.toolBar = QtWidgets.QToolBar(Segmentation)
        self.toolBar.setObjectName("toolBar")
        Segmentation.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpen = QtWidgets.QAction(Segmentation)
        self.actionOpen.setObjectName("actionOpen")
        self.actionRotate = QtWidgets.QAction(Segmentation)
        self.actionRotate.setObjectName("actionRotate")
        self.actionDetect = QtWidgets.QAction(Segmentation)
        self.actionDetect.setObjectName("actionDetect")
        self.actionImbibition = QtWidgets.QAction(Segmentation)
        self.actionImbibition.setObjectName("actionImbibition")
        self.actionSave = QtWidgets.QAction(Segmentation)
        self.actionSave.setObjectName("actionSave")
        self.actionDisplay_xy_slices = QtWidgets.QAction(Segmentation)
        self.actionDisplay_xy_slices.setObjectName("actionDisplay_xy_slices")
        self.actionDisplay_xz_slices = QtWidgets.QAction(Segmentation)
        self.actionDisplay_xz_slices.setObjectName("actionDisplay_xz_slices")
        self.actionDisplay_yz_slices = QtWidgets.QAction(Segmentation)
        self.actionDisplay_yz_slices.setObjectName("actionDisplay_yz_slices")
        self.actionShow_slices = QtWidgets.QAction(Segmentation)
        self.actionShow_slices.setObjectName("actionShow_slices")
        self.actionShow_Detected_colum_view = QtWidgets.QAction(Segmentation)
        self.actionShow_Detected_colum_view.setObjectName("actionShow_Detected_colum_view")
        self.actionShow_columnwise_imbibition_front = QtWidgets.QAction(Segmentation)
        self.actionShow_columnwise_imbibition_front.setObjectName("actionShow_columnwise_imbibition_front")
        self.actionSlice_tensor = QtWidgets.QAction(Segmentation)
        self.actionSlice_tensor.setObjectName("actionSlice_tensor")
        self.actionOpen_Tensor = QtWidgets.QAction(Segmentation)
        self.actionOpen_Tensor.setObjectName("actionOpen_Tensor")
        self.menuOpen.addAction(self.actionOpen)
        self.menuOpen.addAction(self.actionOpen_Tensor)
        self.menuOpen.addSeparator()
        self.menuOpen.addSeparator()
        self.menuOpen.addAction(self.actionSave)
        self.menuView.addAction(self.actionDisplay_xz_slices)
        self.menuView.addAction(self.actionDisplay_yz_slices)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionShow_slices)
        self.menubar.addAction(self.menuOpen.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.toolBar.addAction(self.actionSlice_tensor)
        self.toolBar.addAction(self.actionRotate)

        self.retranslateUi(Segmentation)
        self.display_stack.setCurrentIndex(0)
        self.tools.setCurrentIndex(1)
        self.loading_info_stack.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Segmentation)

    def retranslateUi(self, Segmentation):
        _translate = QtCore.QCoreApplication.translate
        Segmentation.setWindowTitle(_translate("Segmentation", "Micro Tomography Analyzer"))
        self.previous_display_page.setText(_translate("Segmentation", "<"))
        self.next_display_page.setText(_translate("Segmentation", ">"))
        self.group_slicing.setTitle(_translate("Segmentation", "Slice tensor:"))
        self.label_2.setText(_translate("Segmentation", "Slice along first axis:"))
        self.label_3.setText(_translate("Segmentation", "Slice along second axis:"))
        self.reset_slicing.setText(_translate("Segmentation", "Reset preview"))
        self.apply_slicing.setText(_translate("Segmentation", "Recompute tensor"))
        self.groupBox_2.setTitle(_translate("Segmentation", "Rotate tensor:"))
        self.preview_rotation.setText(_translate("Segmentation", "Preview for current slice"))
        self.apply_rotation.setText(_translate("Segmentation", "Recompute tensor"))
        self.groupBox_3.setTitle(_translate("Segmentation", "Select axis:"))
        self.label.setText(_translate("Segmentation", "Slice dimensions"))
        self.axis.setItemText(0, _translate("Segmentation", "yx"))
        self.axis.setItemText(1, _translate("Segmentation", "xz"))
        self.axis.setItemText(2, _translate("Segmentation", "yz"))
        self.file_info.setText(_translate("Segmentation", "No file loaded ..."))
        self.previous.setText(_translate("Segmentation", "<<"))
        self.next.setText(_translate("Segmentation", ">>"))
        self.menuOpen.setTitle(_translate("Segmentation", "Start"))
        self.menuView.setTitle(_translate("Segmentation", "View"))
        self.toolBar.setWindowTitle(_translate("Segmentation", "toolBar"))
        self.actionOpen.setText(_translate("Segmentation", "Open source files"))
        self.actionRotate.setText(_translate("Segmentation", "Rotate tensor"))
        self.actionDetect.setText(_translate("Segmentation", "Detect columns"))
        self.actionImbibition.setText(_translate("Segmentation", "Find imbibition front"))
        self.actionSave.setText(_translate("Segmentation", "Save current tensor"))
        self.actionDisplay_xy_slices.setText(_translate("Segmentation", "Display xy slices"))
        self.actionDisplay_xz_slices.setText(_translate("Segmentation", "Display xz slices"))
        self.actionDisplay_yz_slices.setText(_translate("Segmentation", "Display yz slices"))
        self.actionShow_slices.setText(_translate("Segmentation", "Show slices"))
        self.actionShow_Detected_colum_view.setText(_translate("Segmentation", "Show detected colum view"))
        self.actionShow_columnwise_imbibition_front.setText(_translate("Segmentation", "Show columnwise imbibition front"))
        self.actionSlice_tensor.setText(_translate("Segmentation", "Slice tensor"))
        self.actionOpen_Tensor.setText(_translate("Segmentation", "Open Tensor"))

from imagedisplay import ImageDisplay
