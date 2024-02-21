from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from buttons import CalculateCheckBoxButton, CalculateSpinButton, ChooseButton


class ChooseLayout(QVBoxLayout):
    """Макет для кнопок Выбор"""

    def __init__(self):
        QVBoxLayout.__init__(self)

        self.addWidget(ChooseButton())
        self.addWidget(ChooseButton())
        self.addWidget(ChooseButton())


class ChooseLabelLayout(QVBoxLayout):
    """Макет для подписей к кнопкам Выбор"""

    def __init__(self):
        QVBoxLayout.__init__(self)

        self.addWidget(QLabel("Выбрать МЭР"))
        self.addWidget(QLabel("Выбрать тех.режим по добывающим скважинам"))
        self.addWidget(QLabel("Выбрать тех.режим по нагнетательным скважинам"))


class UploadFilesLayout(QHBoxLayout):
    """Макет для области загрузки файлов"""

    def __init__(self):
        QHBoxLayout.__init__(self)

        self.addLayout(ChooseLabelLayout())
        self.addLayout(ChooseLayout())


class CalculateLabelLayout(QVBoxLayout):
    """Макет для подписей к кнопкам параметров расчета"""

    def __init__(self):
        QVBoxLayout.__init__(self)

        self.addWidget(QLabel("Взять в расчет только те скважины, которые работали последние N месяцев"))
        self.addWidget(QLabel("Расчет с учетом скважин, находившихся в работе за последний год"))


class CalculateButtonLayout(QVBoxLayout):
    """Макет для кнопок в параметрах расчета"""

    def __init__(self):
        QVBoxLayout.__init__(self)

        self.addWidget(CalculateSpinButton())
        self.addWidget(CalculateCheckBoxButton())


class CalculateSettingsLayout(QHBoxLayout):
    """Макет для области параметров расчета"""

    def __init__(self):
        QHBoxLayout.__init__(self)

        self.addLayout(CalculateButtonLayout())
        self.addLayout(CalculateLabelLayout())


class MainLayout(QVBoxLayout):
    """Макет основной страницы"""

    def __init__(self):
        QVBoxLayout.__init__(self)

        self.addWidget(QLabel("Загрузка данных"))
        self.addLayout(UploadFilesLayout())
        self.addWidget(QLabel("Параметры расчета"))
        self.addLayout(CalculateSettingsLayout())

