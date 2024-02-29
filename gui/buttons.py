from PySide6.QtWidgets import QCheckBox, QPushButton, QSpinBox


class CalculateButton(QPushButton):
    """Кнопка расчет"""

    def __init__(self):
        QPushButton.__init__(self)
        # Настройка
        self.setText("Рассчитать")
        self.setFixedSize(100, 40)


class CalculateSpinButton(QSpinBox):
    """Кнопка установки месяцев работы"""

    def __init__(self):
        QSpinBox.__init__(self)


class CalculateCheckBoxButton(QCheckBox):
    """Кнопка для учета скважин, находившихся в работе последний год"""

    def __init__(self):
        QCheckBox.__init__(self)
