import sys

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QMainWindow, QWidget

from .layouts import MainPageGridLayout
from .start_page_elements import OutputConsole, StreamOutput


class MainWindow(QMainWindow):
    """Основное окно приложения"""

    def __init__(self):
        QMainWindow.__init__(self)

        self.widget = QWidget()

        # Настройка окна
        self.setWindowTitle("Модуль расчета рентабельности ППД")

        # Инициализация окна с выходными результатами
        self.output_console = OutputConsole()
        sys.stdout = StreamOutput(text_written=self.on_update_text)

        # Инициализация сетки
        self.grid_layout = MainPageGridLayout()
        self.grid_layout.addWidget(self.output_console, 18, 0, 1, 3)

        self.widget.setLayout(self.grid_layout)
        self.setCentralWidget(self.widget)

        self.show()

    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

    def on_update_text(self, text):
        """Write console output to text widget."""
        cursor = self.output_console.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.output_console.setTextCursor(cursor)
        self.output_console.ensureCursorVisible()
