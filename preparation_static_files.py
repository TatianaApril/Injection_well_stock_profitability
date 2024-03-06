import os
import pandas as pd
import sys

from loguru import logger
from pydantic import ValidationError

from .Schema import dict_coord_column, dict_HHT_column, Validator_Coord, Validator_HHT
from .Utility_function import df_Coordinates_prepare


def check_is_static_files_exists(app_dir_path: str, oilfield_name: str) -> tuple:
    """Проверка, что данные (координаты и толщины) по необходимому месторождению существуют"""

    coordinates_data_path = f"{app_dir_path}\\data\\Координаты"
    coordinates_data_on_field = f"{app_dir_path}\\data\\Координаты\\{oilfield_name}.xlsx"
    if not os.path.isdir(coordinates_data_path):
        logger.warning("Координаты по месторождениям не загружены")
        sys.exit()
    if not os.path.isfile(coordinates_data_on_field):
        logger.warning(f"Координаты по месторождению - {oilfield_name} не загружены")
        sys.exit()

    thickness_data_path = f"{app_dir_path}\\data\\Толщины"
    thickness_data_on_field = f"{app_dir_path}\\data\\Толщины\\{oilfield_name}"

    if not os.path.isdir(thickness_data_path):
        logger.warning("Данные о толщинах по месторождениям не загружены")
        sys.exit()

    if not os.path.isdir(thickness_data_on_field) or len(os.listdir(thickness_data_on_field)) == 0:
        logger.warning(f"Отсутствуют данные о толщинах, месторождение - {oilfield_name}")
        sys.exit()

    return coordinates_data_on_field, thickness_data_on_field


def prepare_coordinates(coordinates_df: pd.DataFrame, min_length_horizont_well: int) -> tuple:
    """Подготовка, валидация и обработка фрейма с данными о координатах по скважинам"""

    coordinates_df_copy = coordinates_df
    coordinates_df_copy.columns = dict_coord_column.values()

    try:
        Validator_Coord(df_dict=coordinates_df_copy.to_dict(orient="records"))
    except ValidationError as e:
        print(e)

    reservoir = coordinates_df_copy.Reservoir_name.unique()

    if len(reservoir) > 1:
        raise ValueError(f"Non-unique field name: {reservoir}")
    else:
        reservoir = reservoir[0]

    coordinates_df_copy = df_Coordinates_prepare(coordinates_df_copy, min_length_horizont_well)
    coordinates_df_copy = coordinates_df_copy.reset_index(drop=True)
    coordinates_df_copy["Well_number"] = coordinates_df_copy["Well_number"].astype(str)

    return coordinates_df_copy, reservoir


def prepare_thickness(path_to_thickness: str) -> pd.DataFrame:
    """Подготовка файлов с толщинами пластов по месторождению"""

    df_thickness = pd.DataFrame()
    for file in os.listdir(path=path_to_thickness):
        name_horizon = file.replace('.xlsx', '')
        logger.info(f"load object: {name_horizon}")
        df = pd.read_excel(path_to_thickness + f"\\{file}", header=1).fillna(0.1)
        df.columns = dict_HHT_column.values()
        df['Horizon'] = name_horizon
        df_thickness = pd.concat([df_thickness,df], sort=False)

    df_thickness = df_thickness.reset_index(drop=True)

    try:
        Validator_HHT(df_dict=df_thickness.to_dict(orient="records"))
    except ValidationError as e:
        print(e)

    return df_thickness
