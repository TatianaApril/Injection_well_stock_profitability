from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton

from buttons import CalculateCheckBoxButton, CalculateSpinButton, CalculateButton, ChooseButton
from utils import CustomLabels, HorizontalLine


class MainPageGridLayout(QGridLayout):
    """Макет основной страницы"""

    def __init__(self):
        QGridLayout.__init__(self)

        test = QPushButton("Рассчитать")
        test.setFixedSize(100, 40)

        # Область загрузки данных
        self.addWidget(CustomLabels("Загрузка данных", mode="bold"), 0, 0)
        self.addWidget(HorizontalLine(), 1, 0, 1, 3)
        self.addWidget(QLabel("Выбрать МЭР"), 2, 0)
        self.addWidget(ChooseButton(), 2, 2)
        self.addWidget(QLabel("Выбрать тех.режим по добывающим скважинам"), 3, 0)
        self.addWidget(ChooseButton(), 3, 2)
        self.addWidget(QLabel("Выбрать тех.режим по нагнетательным скважинам"), 4, 0)
        self.addWidget(ChooseButton(), 4, 2)
        self.addWidget(HorizontalLine(), 5, 0, 1, 3)

        # Область настройки расчета
        self.addWidget(CustomLabels("Параметры расчета", mode="bold"), 6, 0)
        self.addWidget(HorizontalLine(), 7, 0, 1, 3)
        self.addWidget(QLabel("Расчет с учетом скважин, находившихся в работе за последний год"), 8, 0)
        self.addWidget(CalculateSpinButton(), 8, 2)
        self.addWidget(QLabel("Взять в расчет только те скважины, которые работали последние N месяцев"), 9, 0)
        self.addWidget(CalculateCheckBoxButton(), 9, 2)
        self.addWidget(HorizontalLine(), 10, 0, 1, 3)

        # Кнопка рассчет
        self.addWidget(CalculateButton(), 11, 0)

        # Добавть растягивающиеся элементы
        self.setColumnStretch(1, 10)
        self.setRowStretch(12, 10)
