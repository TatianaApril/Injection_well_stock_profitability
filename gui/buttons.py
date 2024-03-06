from PySide6.QtWidgets import QCheckBox, QPushButton, QSpinBox, QDoubleSpinBox


class NNTValueBtn(QDoubleSpinBox):
    """Кнопка задает значение толщины ННТ в метрах"""
    def __init__(self):
        QDoubleSpinBox.__init__(self)
        self.current_value = 0.1
        self.setMinimum(0.1)
        self.setSuffix(" м")
        self.valueChanged.connect(self.set_value)

    def set_value(self, f: float):
        self.current_value = f


class MaxDistanceBtn(QSpinBox):
    """Кнопка задает максимальное расстояние по умолчанию от нагнетательной до реагирующей скважин"""
    def __init__(self):
        QSpinBox.__init__(self)
        self.max_distance = 1_000
        self.setMinimum(1)
        self.setMaximum(10_000)
        self.setValue(self.max_distance)
        self.setSuffix(" м.")
        self.valueChanged.connect(self.set_value)

    def set_value(self, i: int):
        self.max_distance = i


class MinLengthHorizonWellBtn(QSpinBox):
    """Кнопка установка минимальной длины между точками Т1 и Т3, чтобы считать скважину горизонтальной"""
    def __init__(self):
        QSpinBox.__init__(self)
        self.min_horizon_well_length = 150
        self.setMinimum(1)
        self.setMaximum(10_000)
        self.setValue(self.min_horizon_well_length)
        self.setSuffix(" м.")
        self.valueChanged.connect(self.set_value)

    def set_value(self, i: int):
        self.min_horizon_well_length = i


class TimeWorkMinBtn(QSpinBox):
    """Кнопка минимального времени работы скважины в месяц, дней"""
    def __init__(self):
        QSpinBox.__init__(self)
        self.minimum_work_time = 0
        self.setMinimum(0)
        self.setValue(self.minimum_work_time)
        self.setSuffix(" дн.")
        self.valueChanged.connect(self.set_value)

    def set_value(self, i: int):
        self.minimum_work_time = i


class WellOperatingForTheLastYear(QCheckBox):
    """Кнопка выбора учета скважин, находившихся в работе за последний год"""
    def __init__(self):
        QCheckBox.__init__(self)
        self.state = False
        self.stateChanged.connect(self.set_state)

    def set_state(self, s):
        self.state = bool(s)


class MonthOfWorkBtn(QSpinBox):
    """Кнопка установки последних N месяцев работы. По умолчанию учитываются скважины, работающие на дату оценки"""

    def __init__(self):
        QSpinBox.__init__(self)
        self.month_of_work = 0
        self.setMinimum(0)
        self.setSuffix(" мес.")
        self.valueChanged.connect(self.set_value)

    def set_value(self, i: int):
        self.month_of_work = i


class CalculateButton(QPushButton):
    """Кнопка расчет"""

    def __init__(self):
        QPushButton.__init__(self)
        # Настройка
        self.setText("Рассчитать")
        self.setFixedSize(100, 40)
