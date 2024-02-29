import sys

from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel
from PySide6.QtGui import QTextCursor

from .buttons import CalculateCheckBoxButton, CalculateSpinButton, CalculateButton
from .start_page_elements import ChooseBlock, OutputConsole, StreamOutput
from .utils import CustomLabels, HorizontalLine

from ..config import LIST_OF_FIELDS
from ..Main import main_script
from ..preparation_data import check_is_files_exists


class MainPageGridLayout(QGridLayout):
    """Макет основной страницы"""

    def __init__(self, parent=None, **kwargs):
        QGridLayout.__init__(self)

        # Список месторождений для выбора
        self.list_of_fields = QComboBox()
        self.list_of_fields.addItems(LIST_OF_FIELDS)
        self.list_of_fields.setCurrentIndex(-1)

        # Линия для ввода текста - пути файла
        self.choose_mer = ChooseBlock("Укажите путь к файлу МЭР", "Выбрать")
        self.choose_prod = ChooseBlock("Укажите путь к файлу тех. режима по добывающим скважинам", "Выбрать")
        self.choose_inj = ChooseBlock("Укажите путь к файлу тех. режима по нагнетательным", "Выбрать")

        # Кнопка рассчет
        self.calculate_btn = CalculateButton()
        self.calculate_btn.clicked.connect(self.start_calculating)

        # Область выбора месторождения
        self.addWidget(CustomLabels("Выберите месторождение", mode="bold"), 0, 0, 1, 3)
        self.addWidget(HorizontalLine(), 1, 0, 1, 3)
        self.addWidget(self.list_of_fields, 2, 0, 1, 3)

        # Область загрузки данных
        self.addWidget(CustomLabels("Загрузка данных", mode="bold"), 3, 0)
        self.addWidget(HorizontalLine(), 4, 0, 1, 3)
        self.addWidget(self.choose_mer.line_text, 5, 0)
        self.addWidget(self.choose_mer.button, 5, 1)
        self.addWidget(self.choose_prod.line_text, 6, 0)
        self.addWidget(self.choose_prod.button, 6, 1)
        self.addWidget(self.choose_inj.line_text, 7, 0)
        self.addWidget(self.choose_inj.button, 7, 1)
        self.addWidget(HorizontalLine(), 8, 0, 1, 3)

        # Область настройки расчета
        self.addWidget(CustomLabels("Параметры расчета", mode="bold"), 9, 0)
        self.addWidget(HorizontalLine(), 10, 0, 1, 3)
        self.addWidget(QLabel("Расчет с учетом скважин, находившихся в работе за последний год"), 11, 0)
        self.addWidget(CalculateSpinButton(), 11, 1)
        self.addWidget(QLabel("Взять в расчет только те скважины, которые работали последние N месяцев"), 12, 0)
        self.addWidget(CalculateCheckBoxButton(), 12, 1)
        self.addWidget(HorizontalLine(), 13, 0, 1, 3)

        # Кнопка рассчет
        self.addWidget(self.calculate_btn, 14, 0)

        # Добавть растягивающиеся элементы
        # self.setColumnStretch(2, 10)
        # self.setRowStretch(16, 10)

    def start_calculating(self):
        mer_path = self.choose_mer.line_text.text()
        prod_path = self.choose_prod.line_text.text()
        inj_path = self.choose_inj.line_text.text()
        check_is_files_exists(mer_path, prod_path, inj_path)

        main_script(self.list_of_fields.currentText(), mer_path, prod_path, inj_path)


