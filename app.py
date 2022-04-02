from decimal import Decimal
import sys
from PyQt5.QtWidgets import (QFileDialog, QApplication, QComboBox, QDialog,
                             QDialogButtonBox, QGridLayout, QGroupBox,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMenu, QMenuBar, QPushButton, QSpinBox,
                             QTextEdit, QVBoxLayout, QWidget)
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QPixmap
from algo import resize


class Dialog(QDialog):

    def __init__(self):
        super().__init__()

        self.create_main()
        self.create_source_selector()
        self.create_target_display()

        main_layout = QHBoxLayout()

        main_layout.setMenuBar(self._menu_bar)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self._source_selector)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self._target_display)
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)
        self.setWindowTitle("Image Resizer")

    def create_main(self):
        self._menu_bar = QMenuBar()

        self._file_menu = QMenu("&File", self)
        self._exit_action = self._file_menu.addAction("E&xit")
        self._menu_bar.addMenu(self._file_menu)

        self._exit_action.triggered.connect(self.accept)

    def create_source_selector(self):
        self._source_selector = QGroupBox("Source Image")
        layout = QFormLayout()
        file_select_button = QPushButton(f"Select a file...")
        file_select_button.clicked.connect(self.getfile)
        layout.addWidget(file_select_button)
        self.width_box = QSpinBox()
        self.width_box.setValue(80)
        self.height_box = QSpinBox()
        self.height_box.setValue(100)
        choices = QComboBox()
        choices.addItems({"Cols First", "Alternate", "Smart Solve"})
        #  layout.addRow(QLabel("Method:"), choices)
        layout.addRow(QLabel("Width (%):"), self.width_box)
        # layout.addRow(QLabel("Height (%):"), self.height_box)
        run_algo_button = QPushButton(f"Resize Image")
        run_algo_button.clicked.connect(self.resize_image)
        layout.addWidget(run_algo_button)
        self._source_selector.setLayout(layout)

    def create_target_display(self):
        self._target_display = QGroupBox("Images")
        layout = QFormLayout()
        self.source_image = QLabel()
        self.target_image = QLabel()
        layout.addWidget(self.source_image)
        layout.addWidget(self.target_image)
        self._target_display.setLayout(layout)

    def getfile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.src, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "./images/", "All Files (*);;png Files (*.png)", options=options)
        src_img = QPixmap(self.src)
        src_img = src_img.scaledToHeight(400)
        self.source_image.setPixmap(src_img)
        self.source_image.update()

    def resize_image(self):
        tgt_width = int(self.width_box.value())
        self.tgt = resize(self.src, tgt_width)
        tgt_img = QPixmap(self.tgt)
        tgt_img = tgt_img.scaledToHeight(400)
        self.target_image.setPixmap(tgt_img)
        self.target_image.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = Dialog()
    geom = QRect(50, 50, 1200, 800)
    dialog.setGeometry(geom)
    sys.exit(dialog.exec())
