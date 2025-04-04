# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwin.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2080, 840)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon/牙齿.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 730, 140, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(490, 730, 140, 40))
        self.pushButton_2.setObjectName("pushButton_2")
        self.xlabel = QtWidgets.QLabel(self.centralwidget)
        self.xlabel.setGeometry(QtCore.QRect(1110, 690, 100, 30))
        self.xlabel.setObjectName("xlabel")
        self.ylabel = QtWidgets.QLabel(self.centralwidget)
        self.ylabel.setGeometry(QtCore.QRect(1310, 690, 100, 30))
        self.ylabel.setObjectName("ylabel")
        self.dptlabel = QtWidgets.QLabel(self.centralwidget)
        self.dptlabel.setGeometry(QtCore.QRect(1480, 690, 100, 30))
        self.dptlabel.setObjectName("dptlabel")
        self.label_filename = QtWidgets.QLabel(self.centralwidget)
        self.label_filename.setGeometry(QtCore.QRect(1110, 750, 960, 30))
        self.label_filename.setObjectName("label_filename")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(1139, 19, 891, 661))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 2080, 22))
        self.menuBar.setObjectName("menuBar")
        self.menu = QtWidgets.QMenu(self.menuBar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menuBar)
        self.menu_2.setObjectName("menu_2")
        self.menu_4 = QtWidgets.QMenu(self.menuBar)
        self.menu_4.setObjectName("menu_4")
        self.menu_3 = QtWidgets.QMenu(self.menuBar)
        self.menu_3.setObjectName("menu_3")
        MainWindow.setMenuBar(self.menuBar)
        self.actionopen = QtWidgets.QAction(MainWindow)
        self.actionopen.setObjectName("actionopen")
        self.actionsave = QtWidgets.QAction(MainWindow)
        self.actionsave.setObjectName("actionsave")
        self.cameraopen = QtWidgets.QAction(MainWindow)
        self.cameraopen.setObjectName("cameraopen")
        self.cameraclose = QtWidgets.QAction(MainWindow)
        self.cameraclose.setObjectName("cameraclose")
        self.anast = QtWidgets.QAction(MainWindow)
        self.anast.setObjectName("anast")
        self.anaclose = QtWidgets.QAction(MainWindow)
        self.anaclose.setObjectName("anaclose")
        self.anasave = QtWidgets.QAction(MainWindow)
        self.anasave.setObjectName("anasave")
        self.helpaction = QtWidgets.QAction(MainWindow)
        self.helpaction.setObjectName("helpaction")
        self.menu.addAction(self.actionopen)
        self.menu.addAction(self.actionsave)
        self.menu_2.addAction(self.anast)
        self.menu_2.addAction(self.anaclose)
        self.menu_2.addAction(self.anasave)
        self.menu_4.addAction(self.cameraopen)
        self.menu_4.addAction(self.cameraclose)
        self.menu_3.addAction(self.helpaction)
        self.menuBar.addAction(self.menu.menuAction())
        self.menuBar.addAction(self.menu_4.menuAction())
        self.menuBar.addAction(self.menu_2.menuAction())
        self.menuBar.addAction(self.menu_3.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ToothTool"))
        self.pushButton.setText(_translate("MainWindow", "开始/暂停"))
        self.pushButton_2.setText(_translate("MainWindow", "结束"))
        self.xlabel.setText(_translate("MainWindow", "TextLabel"))
        self.ylabel.setText(_translate("MainWindow", "TextLabel"))
        self.dptlabel.setText(_translate("MainWindow", "TextLabel"))
        self.label_filename.setText(_translate("MainWindow", "TextLabel"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "分析"))
        self.menu_4.setTitle(_translate("MainWindow", "摄像头"))
        self.menu_3.setTitle(_translate("MainWindow", "其他"))
        self.actionopen.setText(_translate("MainWindow", "打开"))
        self.actionsave.setText(_translate("MainWindow", "保存"))
        self.cameraopen.setText(_translate("MainWindow", "打开"))
        self.cameraclose.setText(_translate("MainWindow", "关闭"))
        self.anast.setText(_translate("MainWindow", "开始/暂停"))
        self.anaclose.setText(_translate("MainWindow", "结束"))
        self.anasave.setText(_translate("MainWindow", "保存"))
        self.helpaction.setText(_translate("MainWindow", "帮助"))
