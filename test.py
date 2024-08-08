if True:
    from reset_random import reset_random

    reset_random()
import os
import pickle
import sys

import cmapy
import numpy as np
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (QWidget, QApplication, QVBoxLayout, QGroupBox, QGridLayout, QLineEdit, QPushButton,
                             QFileDialog, QMessageBox, QLabel, QScrollArea, QDialog, QProgressBar, QComboBox)
import cv2
from tensorflow.python.keras.applications.densenet import preprocess_input as de_pp
from tensorflow.python.keras.applications.inception_v3 import preprocess_input as iv_pp

from feature_fusion import (get_image_to_predict,
                            get_feature, get_feature_image, d201, iv3, d201_shape, iv3_shape, d201_fmm, iv3_fmm)
from preprocess import contrast_enhance, gaussian_filter
from utils import Worker, DATASET


class AerialSceneClassification(QWidget):
    def __init__(self):
        super(AerialSceneClassification, self).__init__()

        self.setWindowTitle('AerialSceneClassification')
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setWindowState(Qt.WindowMaximized)

        self.app_width = QApplication.desktop().availableGeometry().width()
        self.app_height = QApplication.desktop().availableGeometry().height()

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        self.gb_1 = QGroupBox('Input Data')

        self.gb_1.setFixedWidth((self.app_width // 100) * 99)
        self.gb_1.setFixedHeight((self.app_height // 100) * 10)
        self.grid_1 = QGridLayout()
        self.grid_1.setSpacing(10)
        self.gb_1.setLayout(self.grid_1)

        self.ip_le = QLineEdit()
        self.ip_le.setFixedWidth((self.app_width // 100) * 25)
        self.ip_le.setFocusPolicy(Qt.NoFocus)
        self.grid_1.addWidget(self.ip_le, 0, 0)

        self.dt_combo = QComboBox()
        self.dt_combo_items = ['<-- Choose Dataset -->', 'UCM', 'AID']
        self.grid_1.addWidget(self.dt_combo, 0, 1)

        self.ci_pb = QPushButton('Choose Input Image')
        self.ci_pb.clicked.connect(self.choose_input)
        self.grid_1.addWidget(self.ci_pb, 0, 2)

        self.pp_btn = QPushButton('Preprocess')
        self.pp_btn.clicked.connect(self.preprocess_thread)
        self.grid_1.addWidget(self.pp_btn, 0, 3)

        self.fe_btn = QPushButton('InceptionV3 + DenseNet201')
        self.fe_btn.clicked.connect(self.fe_thread)
        self.grid_1.addWidget(self.fe_btn, 0, 4)

        self.cls_btn = QPushButton('Classify Using KELM')
        self.cls_btn.clicked.connect(self.classify_thread)
        self.grid_1.addWidget(self.cls_btn, 0, 6)

        self.gb_2 = QGroupBox('Results')
        self.gb_2.setFixedWidth((self.app_width // 100) * 99)
        self.gb_2.setFixedHeight((self.app_height // 100) * 85)
        self.grid_2_scroll = QScrollArea()
        self.gb_2_v_box = QVBoxLayout()
        self.grid_2_widget = QWidget()
        self.grid_2 = QGridLayout(self.grid_2_widget)
        self.grid_2_scroll.setWidgetResizable(True)
        self.grid_2_scroll.setWidget(self.grid_2_widget)
        self.gb_2_v_box.addWidget(self.grid_2_scroll)
        self.gb_2_v_box.setContentsMargins(0, 0, 0, 0)
        self.gb_2.setLayout(self.gb_2_v_box)

        self.main_layout.addWidget(self.gb_1)
        self.main_layout.addWidget(self.gb_2)
        self.setLayout(self.main_layout)
        self._input_image_path = ''
        self._image_size = ((self.gb_2.height() // 100) * 90, (self.app_width // 100) * 45)
        self.index = 0
        self.load_screen = Loading()
        self.thread_pool = QThreadPool()
        self.pp_data = {}
        self.fe_data = {}
        self.class_ = None
        self.dt_combo.currentIndexChanged.connect(self.dt_changed)
        self.dt_combo.addItems(self.dt_combo_items)
        self.disable()
        self.show()

    def dt_changed(self):
        if self.dt_combo.currentIndex() != 0:
            self.ci_pb.setEnabled(True)
        else:
            self.ci_pb.setEnabled(False)

    def choose_input(self):
        self.reset()
        filter1 = "TIF Files (*.tif);;BMP Files (*.bmp);;PNG Files (*.PNG)"
        filter2 = "JPG Files (*.jpg);;BMP Files (*.bmp);;PNG Files (*.PNG)"
        self._input_image_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Choose Input Image", directory=".",
            options=QFileDialog.DontUseNativeDialog,
            filter=filter1 if self.dt_combo.currentIndex() == 1 else filter2
        )
        if os.path.isfile(self._input_image_path):
            self.ip_le.setText(self._input_image_path)
            self.add_image(self._input_image_path, 'Input Image')
            self.ci_pb.setEnabled(False)
            self.dt_combo.setEnabled(False)
            self.pp_btn.setEnabled(True)
        else:
            self.show_message_box('InputImageError', QMessageBox.Critical, 'Choose valid image?')

    def preprocess_thread(self):
        worker = Worker(self.preprocess_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.preprocess_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()

    def preprocess_runner(self):
        img = cv2.imread(self._input_image_path)
        gf = gaussian_filter(img)
        self.pp_data['Gaussian Filtered'] = gf
        ce = contrast_enhance(gf)
        self.pp_data['Contrast Enhanced'] = ce

    def preprocess_finisher(self):
        for k in self.pp_data:
            cv2.imwrite('x.jpg', self.pp_data[k])
            self.add_image('x.jpg', k)
        self.load_screen.close()
        self.pp_btn.setEnabled(False)
        self.fe_btn.setEnabled(True)

    def fe_thread(self):
        worker = Worker(self.fe_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.fe_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()

    def fe_runner(self):
        d201_im = get_image_to_predict('x.jpg', d201_shape, de_pp)
        iv3_im = get_image_to_predict('x.jpg', iv3_shape, iv_pp)

        d201_fe = get_feature(d201_im, d201)[:224]
        iv3_fe = get_feature(iv3_im, iv3)[:299]
        d201_iv3_fe = [*d201_fe, *iv3_fe]
        self.fe_data['feature'] = d201_iv3_fe

        r101_fm = get_feature_image(d201_im, d201_fmm)
        iv3_fm = get_feature_image(iv3_im, iv3_fmm)

        r101_fm = cv2.resize(r101_fm, (48, 48))
        iv3_fm = cv2.resize(iv3_fm, (48, 48))

        r101_fm = cv2.applyColorMap(r101_fm, cmapy.cmap('viridis_r'))
        iv3_fm = cv2.applyColorMap(iv3_fm, cmapy.cmap('viridis_r'))
        self.fe_data['FeatureMap'] = r101_fm + iv3_fm

    def fe_finisher(self):
        cv2.imwrite('x.jpg', self.fe_data['FeatureMap'])
        self.add_image('x.jpg', 'FeatureMap')
        self.load_screen.close()
        self.fe_btn.setEnabled(False)
        self.cls_btn.setEnabled(True)

    def classify_thread(self):
        worker = Worker(self.classify_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.classify_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()

    def classify_runner(self):
        feature = np.array(self.fe_data['feature'], ndmin=2)
        with open('classifiers/{0}/ss.pkl'.format(self.dt_combo.currentText()), 'rb') as f:
            ss = pickle.load(f)
            feature = ss.transform(feature)
        with open('classifiers/{0}/classifier{1}.pkl'.format(
                self.dt_combo.currentText(), ''
        ), 'rb') as f:
            classifier = pickle.load(f)
            pred = classifier.predict(feature)[0]
        self.class_ = DATASET[self.dt_combo.currentText()][pred].capitalize()

    def classify_finisher(self):
        self.add_image(self._input_image_path, 'Classified As "{0}"'.format(self.class_))
        os.remove('x.jpg')
        self.cls_btn.setEnabled(False)
        self.ci_pb.setEnabled(True)
        self.dt_combo.setEnabled(True)
        self.load_screen.close()

    @staticmethod
    def clear_layout(layout):
        while layout.count() > 0:
            item = layout.takeAt(0)
            if not item:
                continue
            w = item.widget()
            if w:
                w.deleteLater()

    @staticmethod
    def show_message_box(title, icon, msg):
        msg_box = QMessageBox()
        msg_box.setFont(QFont('Fira Code', 10, 1))
        msg_box.setWindowTitle(title)
        msg_box.setText(msg)
        msg_box.setIcon(icon)
        msg_box.setDefaultButton(QMessageBox.Ok)
        msg_box.setWindowModality(Qt.ApplicationModal)
        msg_box.exec_()

    def add_image(self, im_path, title):
        image_lb = QLabel()
        image_lb.setFixedHeight(self._image_size[0])
        image_lb.setFixedWidth(self._image_size[1])
        image_lb.setScaledContents(True)
        image_lb.setStyleSheet('padding-top: 30px;')
        qimg = QImage(im_path)
        pixmap = QPixmap.fromImage(qimg)
        image_lb.setPixmap(pixmap)
        self.grid_2.addWidget(image_lb, 0, self.index, Qt.AlignCenter)
        txt_lb = QLabel(title)
        self.grid_2.addWidget(txt_lb, 1, self.index, Qt.AlignCenter)
        self.index += 1

    def disable(self):
        self.ip_le.clear()
        self.class_ = None
        self._input_image_path = ''
        self.dt_combo.setEnabled(True)
        self.ci_pb.setEnabled(False)
        self.pp_btn.setEnabled(False)
        self.fe_btn.setEnabled(False)
        self.cls_btn.setEnabled(False)
        self.pp_data = {}
        self.fe_data = {}

    def reset(self):
        self.disable()
        self.clear_layout(self.grid_2)


class Loading(QDialog):
    def __init__(self, parent=None):
        super(Loading, self).__init__(parent)
        self.screen_size = app.primaryScreen().size()
        self._width = int(self.screen_size.width() / 100) * 40
        self._height = int(self.screen_size.height() / 100) * 5
        self.setGeometry(0, 0, self._width, self._height)
        x = (self.screen_size.width() - self.width()) / 2
        y = (self.screen_size.height() - self.height()) / 2
        self.move(x, y)
        self.setWindowFlags(Qt.CustomizeWindowHint)
        self.pb = QProgressBar(self)
        self.pb.setFixedWidth(self.width())
        self.pb.setFixedHeight(self.height())
        self.pb.setRange(0, 0)


if __name__ == '__main__':
    app = QApplication([sys.argv])
    app.setStyle('fusion')
    app.setFont(QFont('JetBrains Mono', 10))
    builder = AerialSceneClassification()
    sys.exit(app.exec_())
