import numpy as np
import pandas as pd
import xlwings as xw
import yaml
import os
import sqlite3
from loguru import logger

from I_Cell_calculate import calculation_coefficients, calculation_injCelle
from II_Oil_increment_calculate import calculate_oil_increment
from III_Uncalculated_wells_and_summation_increments import final_adaptation_and_summation
from IV_Forecast_calculate import calculate_forecast
from Utility_function import merging_sheets

from drainage_area import get_properties, calculate_zones
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

# upload parameters
max_overlap_percent, \
angle_verWell, \
angle_horWell_T1, \
angle_horWell_T3, \
time_predict, \
volume_factor, \
Rw = [0, 0, 0, 0, 0, 0, 0]

# Switches
drainage_areas, dynamic_coefficient = [None, None]

# CONSTANT
DEFAULT_HHT = 0.1  # meters
MAX_DISTANCE: int = 1000  # default maximum distance from injection well for reacting wells

if __name__ == '__main__':

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

    for base in database_content:
        reservoir = base.replace(".db", "")
        logger.info(f"Upload database for reservoir: {reservoir}")
        connection = sqlite3.connect(database_path + f'//{base}')

        df_Coordinates = pd.read_sql("SELECT * from coordinates", connection)
        df_inj = pd.read_sql("SELECT * from inj", connection)
        df_prod = pd.read_sql("SELECT * from prod", connection)
        df_HHT = pd.read_sql("SELECT * from HHT", connection)
        df_HHT .replace(to_replace=0, value=DEFAULT_HHT, inplace=True)

        connection.commit()
        connection.close()

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

        logger.info(f"list of horizons for calculation: {list_horizons}")
        for horizon in list_horizons:
            logger.info(f"Start calculation reservoir: {reservoir} horizon: {horizon}")

            # select the history and HHT for this horizon
            df_inj_horizon = df_inj[df_inj.Horizon == horizon]
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
            last_data = pd.Timestamp(np.sort(df_inj_horizon.Date.unique())[-1])

            logger.info("0. Calculate drainage and injection zones for all wells")
            df_drainage_areas = pd.DataFrame()
            if drainage_areas:
                dict_properties = get_properties(actual_reservoir_properties, [horizon])
                df_drainage_areas = calculate_zones(list_wells, list_prod_wells, df_prod_horizon, df_inj_horizon,
                                                    dict_properties, df_Coordinates, dict_HHT, DEFAULT_HHT)

            logger.info("I. Start calculation of injCelle for each inj well")
            df_injCells_horizon = calculation_injCelle(list_inj_wells, df_Coordinates_horizon, df_inj_horizon,
                                                       df_prod_horizon, reservoir_reaction_distance, dict_HHT,
                                                       df_drainage_areas, drainage_areas,
                                                       max_overlap_percent=max_overlap_percent,
                                                       default_distance=MAX_DISTANCE,
                                                       angle_verWell=angle_verWell,
                                                       angle_horWell_T1=angle_horWell_T1,
                                                       angle_horWell_T3=angle_horWell_T3,
                                                       DEFAULT_HHT=DEFAULT_HHT)

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

            dict_df = {f"Ячейки_{horizon}": df_injCells_horizon, f"Прирост доб_{horizon}": df_final_prod_well,
                       f"Прирост наг_{horizon}": df_final_inj_well, f"Прогноз наг_{horizon}": df_forecasts}

            dict_reservoir_df.update(dict_df)

        # финальная обработка словаря перед загрузкой в эксель
        dict_reservoir_df = merging_sheets(df_injCells_horizon, df_forecasts, dict_reservoir_df, conversion_factor)

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
