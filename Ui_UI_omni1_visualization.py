# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'g:\omniglue\omniglue\UI_omni1_visualization.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(956, 729)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("g:\\omniglue\\omniglue\\icon/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        Form.setStyleSheet("background-color: rgb(243,243,243);\n"
"border-color:rgb(144,167,164);")
        self.gridLayout_6 = QtWidgets.QGridLayout(Form)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.sideBar = QtWidgets.QWidget(Form)
        self.sideBar.setMinimumSize(QtCore.QSize(0, 0))
        self.sideBar.setStyleSheet("QWidget{\n"
"    background-color: rgb(251,251,251);\n"
"    border-radius:10px;\n"
"    border-bottom:2px solid rgb(232,232,232);\n"
"    border-left:3px solid rgb(232,232,232);\n"
"}")
        self.sideBar.setObjectName("sideBar")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.sideBar)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.Order1_bar = QtWidgets.QToolButton(self.sideBar)
        self.Order1_bar.setStyleSheet("/* 默认 */\n"
"QToolButton{   \n"
"    border-top: 3px outset transparent;\n"
"    border-bottom: 7px outset transparent;\n"
"    border-right: 3px outset transparent;\n"
"    border-left: 3px outset transparent;\n"
"    min-width: 80px;\n"
"    min-height: 80px;\n"
"    background-color: rgb(251,251,251);\n"
"    color: rgb(44,44,44);\n"
"    font: 11pt \"微软雅黑\";\n"
"}\n"
"\n"
"/* 鼠标悬停 */\n"
"QToolButton:hover{\n"
"    background-color: rgb(246,246,246);\n"
"}\n"
"\n"
"/* 点击和按下 */\n"
"QToolButton:pressed,QToolButton:checked{\n"
"    border-left: 3px outset rgb(58,104,76);\n"
"    background-color: rgb(234,234,234);\n"
"    color:rgb(44,44,44);\n"
"}\n"
"\n"
"\n"
"\n"
"")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("g:\\omniglue\\omniglue\\icon/图表-折线图.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Order1_bar.setIcon(icon1)
        self.Order1_bar.setIconSize(QtCore.QSize(40, 40))
        self.Order1_bar.setCheckable(True)
        self.Order1_bar.setAutoExclusive(True)
        self.Order1_bar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.Order1_bar.setObjectName("Order1_bar")
        self.verticalLayout_10.addWidget(self.Order1_bar)
        self.Order2_bar = QtWidgets.QToolButton(self.sideBar)
        self.Order2_bar.setMinimumSize(QtCore.QSize(86, 90))
        self.Order2_bar.setStyleSheet("/* 默认 */\n"
"QToolButton{   \n"
"    border-top: 3px outset transparent;\n"
"    border-bottom: 7px outset transparent;\n"
"    border-right: 3px outset transparent;\n"
"    border-left: 3px outset transparent;\n"
"    min-width: 80px;\n"
"    min-height: 80px;\n"
"    background-color: rgb(251,251,251);\n"
"    color: rgb(44,44,44);\n"
"    font: 11pt \"微软雅黑\";\n"
"}\n"
"\n"
"/* 鼠标悬停 */\n"
"QToolButton:hover{\n"
"    background-color: rgb(237,237,237);\n"
"}\n"
"\n"
"/* 点击和按下 */\n"
"QToolButton:pressed,QToolButton:checked{\n"
"    border-left: 3px outset rgb(58,104,76);\n"
"    background-color: rgb(234,234,234);\n"
"    color:rgb(44,44,44);\n"
"}\n"
"\n"
"\n"
"\n"
"")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("g:\\omniglue\\omniglue\\icon/可视化.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Order2_bar.setIcon(icon2)
        self.Order2_bar.setIconSize(QtCore.QSize(40, 40))
        self.Order2_bar.setCheckable(True)
        self.Order2_bar.setAutoExclusive(True)
        self.Order2_bar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.Order2_bar.setObjectName("Order2_bar")
        self.verticalLayout_10.addWidget(self.Order2_bar)
        self.Order4_bar = QtWidgets.QToolButton(self.sideBar)
        self.Order4_bar.setMinimumSize(QtCore.QSize(86, 90))
        self.Order4_bar.setStyleSheet("/* 默认 */\n"
"QToolButton{   \n"
"    border-top: 3px outset transparent;\n"
"    border-bottom: 7px outset transparent;\n"
"    border-right: 3px outset transparent;\n"
"    border-left: 3px outset transparent;\n"
"    min-width: 80px;\n"
"    min-height: 80px;\n"
"    background-color: rgb(251,251,251);\n"
"    color: rgb(44,44,44);\n"
"    font: 11pt \"微软雅黑\";\n"
"}\n"
"\n"
"/* 鼠标悬停 */\n"
"QToolButton:hover{\n"
"    background-color: rgb(246,246,246);\n"
"}\n"
"\n"
"/* 点击和按下 */\n"
"QToolButton:pressed,QToolButton:checked{\n"
"    border-left: 3px outset rgb(58,104,76);\n"
"    background-color: rgb(234,234,234);\n"
"    color:rgb(44,44,44);\n"
"}\n"
"\n"
"\n"
"\n"
"")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("g:\\omniglue\\omniglue\\icon/分析.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Order4_bar.setIcon(icon3)
        self.Order4_bar.setIconSize(QtCore.QSize(40, 40))
        self.Order4_bar.setCheckable(True)
        self.Order4_bar.setAutoExclusive(True)
        self.Order4_bar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.Order4_bar.setObjectName("Order4_bar")
        self.verticalLayout_10.addWidget(self.Order4_bar)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_10.addItem(spacerItem)
        self.gridLayout_4.addLayout(self.verticalLayout_10, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.sideBar)
        self.stackedWidget = QtWidgets.QStackedWidget(Form)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.page)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_2 = QtWidgets.QLabel(self.page)
        self.label_2.setMinimumSize(QtCore.QSize(0, 25))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 25))
        self.label_2.setStyleSheet("font: 14pt \"微软雅黑\";\n"
"color: rgb(41,41,41);")
        self.label_2.setObjectName("label_2")
        self.gridLayout_5.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.page)
        self.label_6.setStyleSheet("background-color: rgb(251,251,251);\n"
"border-radius:10px;\n"
"border:1.5px solid rgb(232,232,232)")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.gridLayout_5.addWidget(self.label_6, 1, 0, 1, 1)
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_3 = QtWidgets.QLabel(self.page_2)
        self.label_3.setMinimumSize(QtCore.QSize(0, 25))
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 25))
        self.label_3.setStyleSheet("font: 14pt \"微软雅黑\";\n"
"color: rgb(41,41,41);")
        self.label_3.setObjectName("label_3")
        self.gridLayout_7.addWidget(self.label_3, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.page_2)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setStyleSheet("font: 10pt \"微软雅黑\";\n"
"color: rgb(41,41,41);")
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.frame)
        self.label_7 = QtWidgets.QLabel(self.page_2)
        self.label_7.setMinimumSize(QtCore.QSize(800, 500))
        self.label_7.setStyleSheet("background-color:rgb(251,251,251);\n"
"border-radius:10px;\n"
"border:1.5px solid rgb(231,231,231);")
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.frame_2 = QtWidgets.QFrame(self.page_2)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setStyleSheet("color: rgb(41,41,41);\n"
"font: 14pt \"微软雅黑\";")
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(353, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 0, 1, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.frame_2)
        self.horizontalSlider.setMinimumSize(QtCore.QSize(261, 0))
        self.horizontalSlider.setMaximum(1000)
        self.horizontalSlider.setPageStep(1000)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout_2.addWidget(self.horizontalSlider, 0, 2, 1, 1)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.page_2)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButton = QtWidgets.QPushButton(self.frame_3)
        self.pushButton.setStyleSheet("QPushButton{\n"
"    color: rgb(44,44,44);\n"
"    background-color: rgb(251,251,251);\n"
"    border-top-left-radius: 10px;\n"
"    border-top-right-radius: 10px;\n"
"    border-bottom-left-radius: 10px;\n"
"    border-bottom-right-radius: 10px;\n"
"    font: 11pt \"微软雅黑\";\n"
"    border:1.5px solid rgb(232,232,232);\n"
"}\n"
"QPushButton:hover{\n"
"    background-color:rgb(237,237,237)\n"
"}\n"
"QPushButton:pressed,QPushButton:checked\n"
"{\n"
"    border-left: 3px outset rgb(58,104,76);\n"
"    background-color:rgb(234,234,234)\n"
"}")
        self.pushButton.setText("")
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_3.addWidget(self.pushButton, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.frame_3)
        self.gridLayout_7.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.page_3)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.frame_4 = QtWidgets.QFrame(self.page_3)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.textEdit = QtWidgets.QTextEdit(self.frame_4)
        self.textEdit.setFocusPolicy(QtCore.Qt.NoFocus)
        self.textEdit.setStyleSheet("background-color:rgb(251,251,251);\n"
"border-radius:10px;\n"
"color:rgb(58,104,76);\n"
"border:1.5px solid rgb(231,231,231);\n"
"font: 12pt \"微软雅黑\";")
        self.textEdit.setObjectName("textEdit")
        self.gridLayout_9.addWidget(self.textEdit, 1, 0, 1, 1)
        self.gridLayout_8.addWidget(self.frame_4, 1, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.page_3)
        self.label_5.setStyleSheet("font: 14pt \"微软雅黑\";\n"
"color: rgb(41,41,41);")
        self.label_5.setObjectName("label_5")
        self.gridLayout_8.addWidget(self.label_5, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_3)
        self.horizontalLayout.addWidget(self.stackedWidget)
        self.gridLayout_6.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.stackedWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "慧眼鉴真(可视化匹配)"))
        self.Order1_bar.setText(_translate("Form", "置信度分布"))
        self.Order2_bar.setText(_translate("Form", "可视化匹配"))
        self.Order4_bar.setText(_translate("Form", "置信度分析"))
        self.label_2.setText(_translate("Form", "置信度分布图"))
        self.label_3.setText(_translate("Form", "可视化匹配"))
        self.label_4.setText(_translate("Form", "系统认为正确匹配的文件名称："))
        self.label.setText(_translate("Form", "置信度选择："))
        self.label_5.setText(_translate("Form", "置信度分析"))
