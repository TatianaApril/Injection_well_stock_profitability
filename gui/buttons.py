from PySide6.QtWidgets import QCheckBox, QPushButton, QSpinBox


class CalculateButton(QPushButton):
    """Кнопка расчет"""

    def __init__(self):
        QPushButton.__init__(self)
        # Настройка
        self.setText("Рассчитать")
        self.setFixedSize(100, 40)
        # Свойства
        self.clicked.connect(self.click_button)

    def click_button(self):
        return 'Hello'


class ChooseButton(QPushButton):
    """Кнопка выбрать"""

    def __init__(self):
        QPushButton.__init__(self)
        # Настройка
        self.setText("Выбрать")
        # Свойства
        self.clicked.connect(self.click_button)

    def click_button(self):
        return 'Hello'


class CalculateSpinButton(QSpinBox):
    """Кнопка установки месяцев работы"""

    def __init__(self):
        QSpinBox.__init__(self)


class CalculateCheckBoxButton(QCheckBox):
    """Кнопка для учета скважин, находившихся в работе последний год"""

    def __init__(self):
        QCheckBox.__init__(self)

