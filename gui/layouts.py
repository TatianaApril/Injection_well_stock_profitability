from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel

from .buttons import CalculateButton, MaxDistanceBtn, MinLengthHorizonWellBtn, MonthOfWorkBtn, NNTValueBtn, TimeWorkMinBtn, WellOperatingForTheLastYear
from .start_page_elements import ChooseBlock
from .utils import CustomLabels, HorizontalLine

from ..config import LIST_OF_FIELDS
from ..Main import main_script
from ..preparation_data import check_is_files_exists


class MainPageGridLayout(QGridLayout):
    """Макет основной страницы"""

    def __init__(self, parent=None, **kwargs):
        QGridLayout.__init__(self)

        # Инициализация кнопок
        self.nnt_value = NNTValueBtn()
        self.max_distance = MaxDistanceBtn()
        self.min_length_horizon = MinLengthHorizonWellBtn()
        self.min_time_work = TimeWorkMinBtn()
        self.well_operating_last_year = WellOperatingForTheLastYear()
        self.month_work = MonthOfWorkBtn()
        self.calculate_btn = CalculateButton()
        self.calculate_btn.clicked.connect(self.start_calculating)

        # Список месторождений для выбора
        self.list_of_fields = QComboBox()
        self.list_of_fields.addItems(LIST_OF_FIELDS)
        self.list_of_fields.setCurrentIndex(-1)

        # Линия для ввода текста - пути файла
        self.choose_mer = ChooseBlock("Укажите путь к файлу МЭР", "Выбрать")
        self.choose_prod = ChooseBlock("Укажите путь к файлу тех. режима по добывающим скважинам", "Выбрать")
        self.choose_inj = ChooseBlock("Укажите путь к файлу тех. режима по нагнетательным", "Выбрать")

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
        self.addWidget(QLabel("Значение толщин по умолчанию"), 11, 0)
        self.addWidget(self.nnt_value, 11, 1)
        self.addWidget(QLabel("Максимальное расстояние от нагнетательной до реагирующей скважин"), 12, 0)
        self.addWidget(self.max_distance, 12, 1)
        self.addWidget(QLabel("Минимальная длина между точками Т1 и Т3"), 13, 0)
        self.addWidget(self.min_length_horizon, 13, 1)
        self.addWidget(QLabel("Минимальное время работы скважины в месяц"), 14, 0)
        self.addWidget(self.min_time_work, 14, 1)
        self.addWidget(QLabel("Расчет с учетом скважин, находившихся в работе за последний год"), 15, 0)
        self.addWidget(self.well_operating_last_year, 15, 1)
        self.addWidget(QLabel("Учитывать скважины, работающие последние N месяцев"), 16, 0)
        self.addWidget(self.month_work, 16, 1)

        # Кнопка рассчет
        self.addWidget(self.calculate_btn, 17, 0)

        # Добавть растягивающиеся элементы
        # self.setColumnStretch(2, 10)
        # self.setRowStretch(16, 10)

    def get_setting_values(self):
        """Считываем настройки с интерфейса перед основным расчетом"""
        default_nnt = self.nnt_value.current_value
        max_distance = self.max_distance
        min_length_horizon = self.min_length_horizon
        min_time_work = self.min_time_work
        well_operating_last_year = self.well_operating_last_year
        month_work = self.month_work

        return default_nnt, max_distance, min_length_horizon, min_time_work, well_operating_last_year, month_work

    def start_calculating(self):
        """Запуск основного скрипта для расчета"""

        # Собираем данные с настроек и передаем в основной скрипт
        default_nnt = self.get_setting_values()

        # Получаем пути к файлам
        mer_path = self.choose_mer.line_text.text()
        prod_path = self.choose_prod.line_text.text()
        inj_path = self.choose_inj.line_text.text()
        check_is_files_exists(mer_path, prod_path, inj_path)

        main_script(self.list_of_fields.currentText(), mer_path, prod_path, inj_path,
                    default_nnt=self.nnt_value.current_value,
                    max_distance=self.max_distance.max_distance,
                    min_length_horizon=self.min_length_horizon.min_horizon_well_length,
                    min_time_work=self.min_time_work.minimum_work_time,
                    well_operating_last_year=self.well_operating_last_year.state,
                    month_work=self.month_work.month_of_work)
