""" Создание базы данных для расчета ячеек на основе файлов из папки input"""
import pandas as pd
import os
from loguru import logger
from Schema import Validator_Coord, Validator_inj, Validator_prod, Validator_HHT
from pydantic import ValidationError
import sqlite3
import warnings

from Utility_function import df_Coordinates_prepare, history_prepare

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

# Parameters
min_length_horWell = 150  # minimum length between points T1 and T3 to consider the well as horizontal
time_work_min = 0  # minimum well's operation time per month, days


dict_prod_column = {'Номер скважины': 'Well_number',
                    'Дата': 'Date',
                    'Состояние на конец месяца': 'Status',
                    'Время работы, ч': 'Time_production_1',
                    'Время работы в периодическом режиме / под циклической закачкой, ч': 'Time_production_2',
                    'Пласт': 'Horizon',
                    'Рзаб, атм': 'Pbh',
                    'KH, мД м': 'Kh',
                    'Pпл, атм': 'Pr',
                    'Qн, т/сут': 'Rate_oil',
                    'Qж, м3/сут': 'Rate_fluid',
                    'Обводненность (объемная), %': 'Water_cut'}

dict_inj_column = {'Номер скважины': 'Well_number',
                   'Дата': 'Date',
                   'Состояние на конец месяца': 'Status',
                   'Диаметр штуцера, мм': 'Choke_size',
                   'Время работы, ч': 'Time_injection_1',
                   'Время работы в периодическом режиме / под циклической закачкой, ч': 'Time_injection_2',
                   'Давление на КНС, атм': "Pkns",
                   'Давление на БГ куста, выкиде насоса, атм': "Pkust",
                   'Давление на устье фактическое, атм': 'Pwh',
                   'Пласт': 'Horizon',
                   'Рбуф': "Pbf",
                   'Рзаб, атм': 'Pbh',
                   'Pпл, атм': 'Pr',
                   'Добыча жидкости/закачка агента для нагнетательных, м3': 'Injection'}

dict_coord_column = {'Меторождение': 'Reservoir_name',
                     '№ скважины': 'Well_number',
                     'Дата': 'Date',
                     'Куст': 'Well_cluster',
                     "Координата X": 'XT1',
                     "Координата Y": 'YT1',
                     "Координата забоя Х (по траектории)": 'XT3',
                     "Координата забоя Y (по траектории)": 'YT3'}

dict_HHT_column = {'Скважина': 'Well_number',
                   'Значение с сетки': 'HHT'}

logger.info("CHECKING FOR FILES")

dir_path = os.path.dirname(os.path.realpath(__file__))
logger.info(f"path:{dir_path}")

database_path = dir_path + "\\database"
database_content = os.listdir(path=database_path)

logger.info("check the content of input")
input_path = dir_path + "\\input"
input_content = os.listdir(path=input_path)
try:
    input_content.remove('Экономика')
except ValueError:
    pass

if input_content:
    logger.info(f"reservoirs: {len(input_content)}")
else:
    raise FileExistsError("no folders!")

for folder in input_content:
    logger.info(f"check the contents of {folder}")
    folder_path = input_path + f"\\{folder}"
    folder_content = os.listdir(path=folder_path)
    if "Техрежим доб.csv" not in os.listdir(path=folder_path):
        raise FileExistsError("Техрежим доб.csv")
    elif "Техрежим наг.csv" not in os.listdir(path=folder_path):
        raise FileExistsError("Техрежим нагн.csv")
    elif "Координаты.xlsx" not in os.listdir(path=folder_path):
        raise FileExistsError("Координаты.xlsx")
    elif "Толщины" not in os.listdir(path=folder_path):
        raise FileExistsError("no folder Толщины")
    else:
        logger.info(f"check the contents of folder Толщины")
        folder_path = folder_path + "\\Толщины"
        folder_content = os.listdir(path=folder_path)
        if folder_content:
            logger.info(f"objects: {len(folder_content)} ")
        else:
            raise FileExistsError("no files!")

logger.info(f"LOAD FILES")
for folder in input_content:
    logger.info(f"load the contents of {folder}")
    folder_path = input_path + f"\\{folder}"
    folder_content = os.listdir(path=folder_path)

    logger.info(f"load Координаты.xlsx")
    df_Coordinates = pd.read_excel(folder_path + "\\Координаты.xlsx")
    df_Coordinates.columns = dict_coord_column.values()

    logger.info(f"validate file")
    try:
        Validator_Coord(df_dict=df_Coordinates.to_dict(orient="records"))
    except ValidationError as e:
        print(e)

    reservoir = df_Coordinates.Reservoir_name.unique()
    if len(reservoir) > 1:
        raise ValueError(f"Non-unique field name: {reservoir}")
    else:
        reservoir = reservoir[0]

    logger.info(f"file preparation")
    df_Coordinates = df_Coordinates_prepare(df_Coordinates, min_length_horWell)

    logger.info(f"load Техрежим наг.csv")
    df_inj = pd.read_csv(folder_path + "\\Техрежим наг.csv", encoding='mbcs', sep=";",
                         index_col=False, decimal=',', low_memory=False).fillna(0)

    # В базе NGT кривая выгрузка - один столбец съехал, поэтому забираем не по названию, а по положению
    # df_inj = df_inj.iloc[:, [0, 1, 2, 6, 15, 16, 17, 18, 19, 23, 30, 32, 33, 36]]
    df_inj = df_inj[list(dict_inj_column.keys())]

    df_inj.columns = dict_inj_column.values()
    df_inj.Date = pd.to_datetime(df_inj.Date, dayfirst=True)

    logger.info(f"validate file")
    try:
        Validator_inj(df_dict=df_inj.to_dict(orient="records"))
    except ValidationError as e:
        print(e)

    logger.info(f"file preparation")
    df_inj = history_prepare(df_inj, type_wells='inj', time_work_min=time_work_min)

    logger.info(f"load Техрежим доб.csv")
    df_prod = pd.read_csv(folder_path + "\\Техрежим доб.csv", encoding='mbcs', sep=";",
                          index_col=False, decimal=',', low_memory=False).fillna(0)

    df_prod = df_prod[list(dict_prod_column.keys())]
    df_prod.columns = dict_prod_column.values()
    df_prod.Date = pd.to_datetime(df_prod.Date, dayfirst=True)

    logger.info(f"validate file")
    try:
        Validator_prod(df_dict=df_prod.to_dict(orient="records"))
    except ValidationError as e:
        print(e)

    logger.info(f"file preparation")
    df_prod = history_prepare(df_prod, type_wells='prod', time_work_min=time_work_min)

    logger.info(f"load Толщины")

    folder_path = folder_path + "\\Толщины"
    folder_content = os.listdir(path=folder_path)
    df_HHT = pd.DataFrame()
    for file in folder_content:
        name_horizon = file.replace('.xlsx', '')
        logger.info(f"load object: {name_horizon}")
        df = pd.read_excel(folder_path + f"\\{file}", header=1).fillna(0)
        df.columns = dict_HHT_column.values()
        df['Horizon'] = name_horizon
        df_HHT = df_HHT.append(df)

    logger.info(f"validate file")
    try:
        Validator_HHT(df_dict=df_HHT.to_dict(orient="records"))
    except ValidationError as e:
        print(e)

    logger.info(f"CREATE BASE: {reservoir}")

    connection = sqlite3.connect(database_path + f'//{reservoir}.db')
    df_Coordinates.to_sql("coordinates", connection, if_exists="replace", index=False)
    df_inj.to_sql("inj", connection, if_exists="replace", index=False)
    df_prod.to_sql("prod", connection, if_exists="replace", index=False)
    df_HHT.to_sql("HHT", connection, if_exists="replace", index=False)
    connection.commit()
    connection.close()

logger.info("good end :)")
