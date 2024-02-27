import numpy as np
import os
import pandas as pd
import warnings
import xlwings as xw
import yaml

from dateutil.relativedelta import relativedelta
from loguru import logger
from pydantic import ValidationError

from I_Cell_calculate import calculation_coefficients, calculation_injCelle
from II_Oil_increment_calculate import calculate_oil_increment
from III_Uncalculated_wells_and_summation_increments import final_adaptation_and_summation
from IV_Forecast_calculate import calculate_forecast
from config import DEFAULT_HHT, MAX_DISTANCE, MONTHS_OF_WORKING, time_work_min
from drainage_area import get_properties, calculate_zones
from preparation_data import (convert_date, final_prepare_data_frames, open_files, prepare_mer_and_prod,
                              prepare_production_data_frame)
from preparation_static_files import check_is_static_files_exists, prepare_coordinates, prepare_thickness
from Schema import dict_inj_column, dict_prod_column, Validator_inj, Validator_prod
from Utility_function import get_period_of_working_for_calculating, history_prepare, merging_sheets
from water_pipeline_facilities import water_pipelines


if __name__ == '__main__':
    # ------------------------------- Предварительная обработка файлов перед расчетом ------------------------------- #

    reservoir = "Узунское"    # Реализовать выбор месторождения в главном окне - радио-кнопка

    warnings.filterwarnings('ignore')
    pd.options.mode.chained_assignment = None  # default='warn'

    logger.info("1. Проверка наличия статических файлов (координаты, толщины)")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logger.info(f"path:{dir_path}")
    coordinates_data_on_field_path, thickness_data_on_field_path = check_is_static_files_exists(dir_path, reservoir)

    logger.info("1.1 Чтение файла с координатами в дата фрейм, подготовка фрейма к работе")
    df_Coordinates = pd.read_excel(coordinates_data_on_field_path)
    df_Coordinates, reservoir = prepare_coordinates(df_Coordinates)

    logger.info("1.2 Чтение файлов с толщинами в дата фрейм, подготовка фрейма к работе")
    df_HHT = prepare_thickness(thickness_data_on_field_path)

    logger.info("2. Подготовка тех. режимов. Восполнение данных о работе скважин из МЭР")
    df_mer, df_prod, df_inj = open_files()
    df_prod = convert_date(df_prod)
    df_inj = convert_date(df_inj)
    df_mer, df_prod = prepare_mer_and_prod(df_mer, df_prod)
    # Делаем слияние по пересекающимся значениям колонок "Скважина", "Дата" и "Пласт"
    df_prod = df_prod.merge(df_mer, how='inner', left_on=['Скважина', 'Дата', 'Пласт'],
                            right_on=['Скважина', 'Дата', 'Пласт'])
    # Заполняем пропущенные значения в колонках "Время работы, ч"
    df_prod['Время работы, ч_x'] = df_prod['Время работы, ч_x'].fillna(df_prod['Время работы, ч_y'])
    # Приводим выходные файлы тех.режима по добывающим и нагнетательным скважинам в итоговый для загрузки вид
    df_prod = prepare_production_data_frame(df_prod)
    df_prod = final_prepare_data_frames(df_prod, mode="prod")
    df_inj = final_prepare_data_frames(df_inj, mode="inj")

    logger.info("3. Обработка тех. режима по нагнетательным скважинам")
    df_inj = df_inj[list(dict_inj_column.keys())]
    df_inj.columns = dict_inj_column.values()
    df_inj.Date = pd.to_datetime(df_inj.Date, dayfirst=True)

    if MONTHS_OF_WORKING:
        df_inj = get_period_of_working_for_calculating(df_inj, MONTHS_OF_WORKING)

    logger.info("3.1 Валидация и обработка файла")

    try:
        Validator_inj(df_dict=df_inj.to_dict(orient="records"))
    except ValidationError as e:
        print(e)

    df_inj = history_prepare(df_inj, type_wells='inj', time_work_min=time_work_min)

    logger.info("4 Обработка тех. режима по добывающим скважинам")
    df_prod = df_prod[df_prod['Способ эксплуатации'] == 'ЭЦН']  # Костыль, фильтровать в препроцессинге
    df_prod = df_prod[list(dict_prod_column.keys())]
    df_prod.columns = dict_prod_column.values()
    df_prod.Date = pd.to_datetime(df_prod.Date, dayfirst=True)

    if MONTHS_OF_WORKING:
        df_prod = get_period_of_working_for_calculating(df_prod, MONTHS_OF_WORKING)

    logger.info(f"4.1 Валидация и обработка файла")
    try:
        Validator_prod(df_dict=df_prod.to_dict(orient="records"))
    except ValidationError as e:
        print(e)

    df_prod = history_prepare(df_prod, type_wells='prod', time_work_min=time_work_min)

    logger.info("Успешно")

    # ------------------------------------- Расчет ячеек

    dir_path = os.path.dirname(os.path.realpath(__file__))
    logger.info(f"path:{dir_path}")

    logger.info(f"upload conf_files")
    with open('conf_files/initial_coefficient.yml') as f:
        initial_coefficient = pd.DataFrame(yaml.safe_load(f))
    with open('conf_files/reservoir_properties.yml', 'rt', encoding='utf8') as yml:
        reservoir_properties = yaml.load(yml, Loader=yaml.Loader)
    with open('conf_files/max_reaction_distance.yml', 'rt', encoding='utf8') as yml:
        max_reaction_distance = yaml.load(yml, Loader=yaml.Loader)
    with open('conf_files/parameters.yml', 'rt', encoding='utf8') as yml:
        parameters = yaml.load(yml, Loader=yaml.Loader)

    max_overlap_percent, \
        angle_verWell, \
        angle_horWell_T1, \
        angle_horWell_T3, \
        time_predict, \
        volume_factor, \
        Rw, \
        drainage_areas, \
        dynamic_coefficient = parameters.values()

    conversion_factor = volume_factor * Rw

    database_path = dir_path + "\\database"
    database_content = os.listdir(path=database_path)

    try:
        database_content.remove('Экономика.db')
    except ValueError:
        pass

    reservoir = df_Coordinates['Reservoir_name'].unique()[0]

    logger.info(f"Upload database for reservoir: {reservoir}")

    df_HHT.replace(to_replace=0, value=DEFAULT_HHT, inplace=True)

    df_inj.Date = pd.to_datetime(df_inj.Date, dayfirst=True)
    df_prod.Date = pd.to_datetime(df_prod.Date, dayfirst=True)

    # upload reservoir_properties
    actual_reservoir_properties = {}
    if drainage_areas:
        actual_reservoir_properties = reservoir_properties.get(reservoir)
        if actual_reservoir_properties is None:
            raise KeyError(f"There is no properties for reservoirs: {reservoir}")

    list_horizons = df_inj.Horizon.unique()

    # set of wells with coordinates
    set_wells = set(df_Coordinates.Well_number.unique())

    # create empty dictionary for result
    dict_reservoir_df = {}

    # Пустой датафрейм для добавления скважин, исключенных из расчета
    df_exception_wells = pd.DataFrame()

    logger.info(f"list of horizons for calculation: {list_horizons}")
    for horizon in list_horizons:
        logger.info(f"Start calculation reservoir: {reservoir} horizon: {horizon}")

        # select the history and HHT for this horizon
        df_inj_horizon = df_inj[df_inj.Horizon == horizon]

        # Считаем количество месяцев работы от даты расчета. Минимально необходимо 6 месяцев, если меньше, то не
        # учитывать нагнетательную скважину в расчете
        date_before_six_month = df_inj_horizon.Date.max() - relativedelta(months=6)
        count_months = df_inj_horizon[df_inj_horizon.Date >= date_before_six_month].groupby(
                       'Well_number', as_index=False).agg({'Date': 'count'})
        df_inj_wells_no_working_six_months = df_inj_horizon[
            df_inj_horizon.Well_number.isin(list(count_months[count_months.Date < 7].Well_number.unique()))]
        df_inj_wells_no_working_six_months.sort_values(by=['Date'], ascending=False, inplace=True)
        df_inj_wells_no_working_six_months = df_inj_wells_no_working_six_months.drop_duplicates(
            subset=['Well_number'])
        df_inj_wells_no_working_six_months['Exception_reason'] = 'последний период работы менее 6 месяцев'
        df_exception_wells = df_exception_wells.append(df_inj_wells_no_working_six_months, ignore_index=True)

        df_inj_horizon = df_inj_horizon[df_inj_horizon.Well_number.isin(
                         list(count_months[count_months.Date >= 7].Well_number.unique()))]
        df_prod_horizon = df_prod[df_prod.Horizon == horizon]
        df_HHT_horizon = df_HHT[df_HHT.Horizon == horizon]
        del df_HHT_horizon["Horizon"]

        # upload dict of effective oil height
        dict_HHT: object = df_HHT_horizon.set_index('Well_number').to_dict('index')

        # create list of inj and prod wells
        list_inj_wells = list(set_wells.intersection(set(df_inj_horizon.Well_number.unique())))
        list_prod_wells = list(set_wells.intersection(set(df_prod_horizon.Well_number.unique())))
        list_wells = list_inj_wells + list_prod_wells

        # leave the intersections with df_Coordinates_horizon
        df_Coordinates_horizon = df_Coordinates[df_Coordinates.Well_number.isin(list_wells)]
        df_Coordinates_horizon["well marker"] = 0
        df_Coordinates_horizon.loc[df_Coordinates_horizon.Well_number.isin(list_inj_wells), "well marker"] = "inj"
        df_Coordinates_horizon.loc[df_Coordinates_horizon.Well_number.isin(list_prod_wells), "well marker"] = "prod"

        # check dictionary for this reservoir
        reservoir_reaction_distance = max_reaction_distance.get(reservoir, {reservoir: None})
        if len(df_inj_horizon.Date.unique()) == 0:
            continue
        last_data = pd.Timestamp(np.sort(df_inj_horizon.Date.unique())[-1])

        logger.info("0. Calculate drainage and injection zones for all wells")
        df_drainage_areas = pd.DataFrame()
        if drainage_areas:
            dict_properties = get_properties(actual_reservoir_properties, [horizon])
            df_drainage_areas = calculate_zones(list_wells, list_prod_wells, df_prod_horizon, df_inj_horizon,
                                                dict_properties, df_Coordinates, dict_HHT, DEFAULT_HHT)

        logger.info("I. Start calculation of injCelle for each inj well")
        df_injCells_horizon, \
            df_inj_wells_without_surrounding = calculation_injCelle(list_inj_wells,
                                                                    df_Coordinates_horizon,
                                                                    df_inj_horizon,
                                                                    df_prod_horizon,
                                                                    reservoir_reaction_distance,
                                                                    dict_HHT,
                                                                    df_drainage_areas,
                                                                    drainage_areas,
                                                                    max_overlap_percent=max_overlap_percent,
                                                                    default_distance=MAX_DISTANCE,
                                                                    angle_verWell=angle_verWell,
                                                                    angle_horWell_T1=angle_horWell_T1,
                                                                    angle_horWell_T3=angle_horWell_T3,
                                                                    DEFAULT_HHT=DEFAULT_HHT)
        df_inj_wells_without_surrounding['Exception_reason'] = 'отсутствует окружение'
        df_exception_wells = df_exception_wells.append(df_inj_wells_without_surrounding, ignore_index=True)
        if df_injCells_horizon.empty:
            continue
        # Sheet "Ячейки"
        df_injCells_horizon = calculation_coefficients(df_injCells_horizon, initial_coefficient,
                                                       dynamic_coefficient)
        list_inj_wells = list(df_injCells_horizon["Ячейка"].unique())

        logger.info("II. Calculate oil increment for each injection well")
        df_final_prod_well, dict_averaged_effects, dict_uncalculated_cells = \
            calculate_oil_increment(df_prod_horizon, last_data, horizon, df_injCells_horizon)

        logger.info("III. Adaptation of uncalculated wells")
        df_final_inj_well, df_final_prod_well = final_adaptation_and_summation(df_prod_horizon, df_inj_horizon,
                                                                               df_final_prod_well, last_data,
                                                                               horizon, df_injCells_horizon,
                                                                               dict_uncalculated_cells,
                                                                               dict_averaged_effects,
                                                                               conversion_factor)
        logger.info("IV. Forecast")
        df_forecasts = calculate_forecast(list_inj_wells, df_final_inj_well, df_injCells_horizon,
                                          horizon, time_predict)
        df_injCells_horizon.insert(3, 'Водовод', 'Нет данных')

        if water_pipelines.get(reservoir):
            for key in water_pipelines[reservoir].keys():
                df_injCells_horizon['Ячейка'] = df_injCells_horizon['Ячейка'].astype(str)
                df_injCells_horizon['Водовод'].loc[df_injCells_horizon['Ячейка'].isin(water_pipelines[reservoir][key])] = key

        dict_df = {f"Ячейки_{horizon}": df_injCells_horizon, f"Прирост доб_{horizon}": df_final_prod_well,
                   f"Прирост наг_{horizon}": df_final_inj_well, f"Прогноз наг_{horizon}": df_forecasts}

        dict_reservoir_df.update(dict_df)

        # dict_reservoir_df.update(df_exception_wells)
        df_exception_wells = df_exception_wells.drop_duplicates(subset=['Well_number'])
        df_exception_wells = df_exception_wells.drop(labels=[
            'Date', 'Status', 'Choke_size', 'Pbh', 'Pkns', 'Pkust', 'Pwh', 'Pbf', 'Pr', 'Time_injection'],
            axis=1).reset_index().drop(labels=['index'], axis=1)

        # финальная обработка словаря перед загрузкой в эксель
        dict_reservoir_df = merging_sheets(df_injCells_horizon, df_forecasts, dict_reservoir_df, df_exception_wells,
                                           conversion_factor)

        # Start print in Excel for one reservoir
        app1 = xw.App(visible=False)
        new_wb = xw.Book()

        for key in dict_reservoir_df.keys():
            if f"{key}" in new_wb.sheets:
                xw.Sheet[f"{key}"].delete()
            new_wb.sheets.add(f"{key}")
            sht = new_wb.sheets(f"{key}")
            sht.range('A1').options().value = dict_reservoir_df[key]

        new_wb.save(dir_path + f"\\output\\{reservoir}.xlsx")
        app1.kill()
