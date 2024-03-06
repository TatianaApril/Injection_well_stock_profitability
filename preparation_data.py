import os
import pandas as pd
import sys

from loguru import logger


def check_is_files_exists(path_to_mer: str, path_to_prod: str, path_to_inj: str) -> None:
    """Проверка, что файлы существуют (МЭР и тех. режимы по добывающим и нагнетательным скважинам"""

    if not os.path.isfile(path_to_mer):
        logger.warning("Неверно указан путь к файлу МЭР")
        raise "Неверно указан путь к файлу МЭР"
    if not os.path.isfile(path_to_prod):
        logger.warning("Неверно указан путь к тех. режиму добывающих скважин")
        raise "Неверно указан путь к тех. режиму добывающих скважин"
    if not os.path.isfile(path_to_inj):
        logger.warning("Неверно указан путь к тех. режиму нагнетательных скважин")
        raise "Неверно указан путь к тех. режиму нагнетательных скважин"

    return


def get_files(path_to_mer: str, path_to_prod: str, path_to_inj: str) -> tuple:
    """Открытие выбранных файлов - МЭР, Тех. режим по добывающим, Тех. режим по нагнетательным"""

    try:
        mer = pd.read_csv(path_to_mer, delimiter=';', encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        mer = pd.read_csv(path_to_mer, delimiter=';', encoding='ANSI', low_memory=False)
    try:
        prod = pd.read_csv(path_to_prod, delimiter=';', encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        prod = pd.read_csv(path_to_prod, delimiter=';', encoding='ANSI', low_memory=False)
    try:
        inj = pd.read_csv(path_to_inj, delimiter=';', encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        inj = pd.read_csv(path_to_inj, delimiter=';', encoding='ANSI', low_memory=False)
    return mer, prod, inj


def convert_month_to_digit_mode(date_as_list):
    """Периводим дату в конвертируемый формат"""
    result = date_as_list
    dict_of_date = {'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04', 'май': '05', 'июн': '06', 'июл': '07',
                    'авг': '08', 'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12'}
    result[1] = dict_of_date[result[1]]


def prepare_production_data_frame(df_prod_wells: pd.DataFrame) -> pd.DataFrame:
    """Переименование колонок тех.режима по добыче в итоговый вид"""

    df_prod_wells = df_prod_wells.rename(columns={'Скважина': 'Номер скважины',
                                                  'Состояние_x': 'Состояние на конец месяца',
                                                  'Время работы, ч_x': 'Время работы, ч', 'Пласт_x': 'Пласт'})
    list_of_columns = ['Номер скважины', 'Куст', 'Дата', 'Состояние на конец месяца', 'Способ эксплуатации',
                       'Рентабельность', 'Внутренний диаметр эксплуатационной колоны, мм', 'Пласт', 'Qн, т/сут',
                       'Qж, м3/сут', 'Обводненность (объемная), %', 'Рзаб, атм', 'Pпл, атм',
                       'Коэф. продуктивности, м3/сут/атм', 'KH, мД м', 'Скин-фактор', 'Радиус контура питания, м',
                       'Динамический уровень, м', 'Буферное давление, атм', 'Pлин, атм', 'Pзатр, атм',
                       'Давление на приеме насоса, атм', 'Статический уровень', 'Рзатр при Нстат, атм', 'Тип насоса',
                       'Дата изм. параметров насоса', 'Глубина спуска насоса, м', 'Номинальный напор ЭЦН, м',
                       'Частота работы ЭЦН, Гц', 'Сила тока ЭЦН, А', 'Номинальная производительность, м3/сут',
                       'Тип пакера', 'Дата установки пакера', 'Глубина установки пакера, м', 'Диаметр штуцера, мм',
                       'В-сть нефти в пластовых условиях, сПз',
                       'Плотность нефти (агента закачки для нагнетательных) в поверхностных условиях',
                       'Объемный коэффициент нефти, м3/м3', 'Замеренный газовый фактор, м3/т',
                       'Глубина верхних дыр перфорации, м', 'Удлинение, м', 'Перфорированная мощность, м',
                       'Нефтенасыщенная (для добывающих) / эффективная (для нагнетательных) толщина, м',
                       'Внешний диаметр НКТ, мм',
                       'Вязкость жидкости (агента закачки для нагнетательных) в поверхностных условиях, сПз',
                       'Добыча нефти, т', 'Добыча жидкости/закачка агента для нагнетательных, м3', 'Время работы, ч',
                       'Время работы в периодическом режиме / под циклической закачкой, ч',
                       'Дебит нефти потенциальный (технологический), т/сут',
                       'Дебит жидкости потенциальный (технологический), м3/сут',
                       'Плотность воды в пластовых условиях, г/см3',
                       'Qж с поправкой на диаметр эксп. колонны (технологический), м3/сут', 'Пуск', 'Остановка',
                       'Проницаемость', 'Тип', 'Примечание']

    df_prod_wells = df_prod_wells[list_of_columns]

    return df_prod_wells


def final_prepare_data_frames(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Финальная подготовка дата фрейма к работе, конвертация форматов ячеек"""

    df = df.drop_duplicates().fillna(0)
    df['Пласт'] = df['Пласт'].replace(list_of_objects_of_field)

    if mode == "prod":
        df["Qн, т/сут"] = df["Qн, т/сут"].str.replace(",", ".").astype(float)
        df["Qж, м3/сут"] = df["Qж, м3/сут"].str.replace(",", ".").astype(float)
        df["Обводненность (объемная), %"] = df["Обводненность (объемная), %"].str.replace(",", ".").astype(float)
        df["Рзаб, атм"] = df["Рзаб, атм"].str.replace(",", ".").astype(float)
        df["Pпл, атм"] = df["Pпл, атм"].str.replace(",", ".").astype(float)
        df["KH, мД м"] = df["KH, мД м"].str.replace(",", ".").astype(float)
    elif mode == "inj":
        df["Рзаб, атм"] = df["Рзаб, атм"].str.replace(",", ".").astype(float)
        df["Pпл, атм"] = df["Pпл, атм"].str.replace(",", ".").astype(float)

    return df


def convert_date(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Приводим дату в конвертируемый формат"""
    df = data_frame
    # Приводим дату в конвертируемый формат во фрейме с тех.режимом добывающих скважин
    df['Дата'] = df['Дата'].str.split('.')
    df['Дата'].map(lambda x: x.insert(0, '01'))
    df['Дата'].apply(convert_month_to_digit_mode)
    df['Дата'] = df['Дата'].map(lambda x: '.'.join(x))

    return df


def prepare_mer_and_prod(mer_df: pd.DataFrame, prod_df: pd.DataFrame) -> tuple:
    """Предварительная обработка фреймов МЭР и тех. режим по добывающим скважинам"""

    mer_df = mer_df.rename(columns={'имя скважины': 'Скважина', 'дата(дд.мм.гггг)': 'Дата',
                                    'время работы': 'Время работы, ч', 'состояние': 'Состояние', 'пласт': 'Пласт'})
    prod_df = prod_df.rename(columns={'Номер скважины': 'Скважина', 'Состояние на конец месяца': 'Состояние'})
    mer_df = mer_df.sort_values(by=['Дата'], ascending=False).reset_index(drop=True)
    prod_df = prod_df.sort_values(by=['Дата'], ascending=False).reset_index(drop=True)

    return mer_df, prod_df


# Словарь для переименовки объектов разработки
list_of_objects_of_field = {'АВ1/3': 'АВ1_3', 'АВ1/3+АВ2/1-2': 'АВ1_3_АВ2_1_2',
                            'ЮВ1/1': 'ЮВ1_1', 'Ю1(1)': 'Ю1_1', 'Ю1/1': 'Ю1_1', 'ЮВ1/2': 'ЮВ1_2', 'ЮС1/1': 'ЮС1_1',
                            'БВ8/1': 'БВ8_1', 'БВ10/2': 'БВ10(2)', 'БС10/1-2': 'БС10_1_2', 'БВ3/1': 'БВ3_1'}
