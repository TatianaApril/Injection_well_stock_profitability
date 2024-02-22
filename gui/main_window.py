from PySide6.QtWidgets import QMainWindow, QWidget

from layouts import MainPageGridLayout


class MainWindow(QMainWindow):
    """Основное окно приложения"""

    def __init__(self):
        QMainWindow.__init__(self)

        widget = QWidget()

        # Настройка окна
        self.setWindowTitle("Модуль расчета рентабельности ППД")
        # self.setMinimumSize(QSize(800, 600))

        widget.setLayout(MainPageGridLayout())

        self.setCentralWidget(widget)
