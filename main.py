import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap
from keras import Model
from keras.src.applications.resnet import ResNet50
from keras.src.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.python.keras.regularizers import l2


from PyQt5 import QtCore, QtGui, QtWidgets
warnings.filterwarnings("ignore")
import tensorflow as tf
import h5py
import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import cv2

import time




model_loaded = keras.models.load_model('ResNet_model.h5')

name_class=['Киста', 'Норма', 'Камни', 'Опухоль']






filename=[]

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 150, 93, 28))
        self.pushButton.setObjectName("pushButton")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(120, 60, 301, 281))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(170, 390, 200, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(170, 390, 200, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(170, 390, 200, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(170, 390, 200, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(170, 390, 200, 16))
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.add_fun()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "KidneyScanAI"))
        self.pushButton.setText(_translate("MainWindow", "Добавить"))



    def add_fun(self):
        self.pushButton.clicked.connect(self.getDirectory)


    def getDirectory(self):

        filename=QFileDialog.getOpenFileName()


        pixmap = QPixmap(filename[0])
        self.label.setGeometry(QtCore.QRect(70, 60, pixmap.width(), pixmap.height()))
        self.label_2.setGeometry(QtCore.QRect(100+pixmap.width(), 310, 200, 16))
        self.label_3.setGeometry(QtCore.QRect(100 + pixmap.width(), 260, 200, 16))
        self.label_4.setGeometry(QtCore.QRect(100 + pixmap.width(), 240, 200, 16))
        self.label_5.setGeometry(QtCore.QRect(100 + pixmap.width(), 220, 200, 16))
        self.label_6.setGeometry(QtCore.QRect(100 + pixmap.width(), 280, 200, 16))
        self.pushButton.setGeometry(QtCore.QRect(100+pixmap.width(), 150, 93, 28))
        self.label.setPixmap(pixmap)

        fn = filename[0].translate(str.maketrans({'/': '\\'}))

        img = cv2.imread(fn)
        resize = tf.image.resize(img, (224,224))
        yhat = model_loaded.predict(np.expand_dims(resize/255, 0))
        max_index = np.argmax(yhat)
        self.label_2.setText("Результат: " +name_class[max_index])
        self.label_3.setText("Норма:    " + str("{:.10f}".format(float(yhat[0][1]))))
        self.label_4.setText("Камни:    " + str("{:.10f}".format(float(yhat[0][2]))))
        self.label_5.setText("Киста:    " + str("{:.10f}".format(float(yhat[0][0]))))
        self.label_6.setText("Опухоль: " + str("{:.10f}".format(float(yhat[0][3]))))








if __name__ == "__main__":

    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()
    sys.exit(app.exec_())





