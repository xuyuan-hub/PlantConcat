import logging
import os
import sys
import cv2
import csv
import shutil
import datetime
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QTableWidgetItem
from PyQt5.QtCore import pyqtSignal,QTimer,QThread
from PyQt5 import QtGui
from ImageProcess import launch as plantLaunch

ROOT = os.path.dirname(os.path.abspath(__file__))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from GUI.PlantConcat import Ui_mainWindow as PlantConcatUi

class QLogBrowserHandler(logging.Handler):
    def __init__(self, text_edit):
        super(QLogBrowserHandler, self).__init__()
        self.text_edit = text_edit
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    def emit(self, record):
        msg = self.format(record)
        self.text_edit.append(msg)
        QApplication.processEvents()

class myMainWindow(PlantConcatUi,QMainWindow):
    _signal = pyqtSignal

    def __init__(self):
        super(PlantConcatUi,self).__init__()
        self.setupUi(self)
        self._init_logger()
        self.slot_init()
        self.valid_images_format = ['jpg', 'tif', 'png', 'jpeg']

    def slot_init(self):
        self.selectDirBtn.clicked.connect(self.open_directory)
        self.startBtn.clicked.connect(self.launch)

    def _init_logger(self):
        self.LOGGER = logging.Logger("Plant Process Tool Logger")
        self.LOGGER.setLevel(logging.DEBUG)
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.run_path = os.path.join(ROOT,'run',formatted_datetime)
        log_browser_handle = QLogBrowserHandler(self.logBrowser)
        self.LOGGER.addHandler(log_browser_handle)

        if not os.path.isdir(self.run_path):
            os.mkdir(self.run_path)
        log_file_path = os.path.join(self.run_path,'run.log')
        filehandle = logging.FileHandler(log_file_path)
        self.LOGGER.addHandler(filehandle)

    def open_directory(self):
        self.selected_dir = QFileDialog.getExistingDirectory(self)
        try:
            self.imagePath = self.selected_dir
            self.input_images_path = []
            self.dirInput.setText(self.imagePath)
            for item in os.listdir(self.selected_dir):
                if os.path.basename(item).split('.')[-1].lower() in self.valid_images_format:
                    self.input_images_path.append(os.path.join(self.selected_dir, item))
            self.num_of_images = len(self.input_images_path)
            self.LOGGER.info(f"Detect there are {self.num_of_images} pictures at {self.selected_dir}")
        except:
            self.LOGGER.info(f"Invalid,Please choose a Directory")

    def launch(self):
        self.LOGGER.info(f"Run!")
        xml_rel_path = self.xmlConfigInput.text()
        xml_abs_path = os.path.join(ROOT,xml_rel_path)
        if os.path.isfile(xml_abs_path):
            self.LOGGER.info(f"Detected crop img Config file at {xml_abs_path}")
            self.result_path = os.path.join(self.run_path,'result')
            os.mkdir(self.result_path)
            # try:
            plantLaunch(xml_path=xml_abs_path,imgs_path=self.imagePath,results_path=self.result_path)
            self.LOGGER.info("Task finished!")
            self.LOGGER.info(f"Result is saved at {self.result_path}")
            # except Exception as e:
            #     self.LOGGER.error(f"Error: {e}")
        else:
            self.LOGGER.error(f"Error: {xml_abs_path} isn't exist or invalid!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller_gui = myMainWindow()
    controller_gui.show()
    sys.exit(app.exec_())