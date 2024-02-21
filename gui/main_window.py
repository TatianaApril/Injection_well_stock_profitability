import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from layouts import MainLayout


class MainWindow(QMainWindow):
    """Основное окно приложения"""

    def __init__(self):
        QMainWindow.__init__(self)

        widget = QWidget()

        # Настройка окна
        self.setWindowTitle("Модуль расчета рентабельности ППД")
        # self.setMinimumSize(QSize(800, 600))

        widget.setLayout(MainLayout())

        self.setCentralWidget(widget)
