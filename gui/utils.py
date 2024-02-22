from PySide6.QtWidgets import QFrame, QLabel


class HorizontalLine(QFrame):
    """Горизонтальная линия"""

    def __init__(self):
        QFrame.__init__(self)

        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)


class CustomLabels(QLabel):
    """Создаем пользовательский текст"""

    def __init__(self, label_name: str, mode: str):
        QLabel.__init__(self)

        self.setText(label_name)

        if mode == "bold":
            self.font = self.font()
            self.font.setBold(True)
            self.setFont(self.font)
        elif mode == "italic":
            pass
        else:
            pass

