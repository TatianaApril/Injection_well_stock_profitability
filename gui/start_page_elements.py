from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QLineEdit, QPushButton, QTextEdit


class ChooseBlock:
    def __init__(self, placeholder_text: str, button_text: str):

        self.line_text = QLineEdit()
        self.line_text.setPlaceholderText(placeholder_text)

        self.button = QPushButton(button_text)
        self.button.clicked.connect(self.get_file_path)

    def get_file_path(self):
        filename, selected_filter = QFileDialog.getOpenFileName(self.button)
        if filename:
            self.line_text.setText(filename)
            self.line_text.setDisabled(True)


class OutputConsole(QTextEdit):
    """Окно отображения процесса расчета"""

    def __init__(self):
        QTextEdit.__init__(self)
        self.setReadOnly(True)


class StreamOutput(QObject):
    text_written = Signal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass
