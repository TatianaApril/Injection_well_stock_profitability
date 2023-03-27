import os

import numpy as np
import pandas as pd
import xlwings as xw
import yaml
from pydantic import ValidationError

from Cell_calculate import cell_definition, calculation_coefficients
from Production_Gain import calculate_production_gain
from Schema import ValidatorMOR
from Utility_function import history_processing
from drainage_area import R_inj, R_prod, get_properties, get_polygon_well
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

# Parameters
maximum_distance: int = 1000  # maximum distance from injection well for reacting wells
min_length_horWell = 150  # minimum length between points T1 and T3 to consider the well as horizontal
max_overlap_percent = 20  # how much one well can cover the sector of another for selection in the first row of wells
angle_verWell = 10  # degree: sector for vertical wells
angle_horWell_T1 = 0  # sector expansion angle for horizontal well's T1 point
angle_horWell_T3 = 0  # sector expansion angle for horizontal well's T3 point
time_predict = 36  # the number of months in the forecast
time_work_min = 5  # minimum well's operation time per month?, days

# for Injection_ratio, %
volume_factor = 1.01  # volume factor of injected fluid
Rw = 1  # density of injected water g/cm3

# Switches
drainage_areas: bool = False
dynamic_coefficient: bool = False

dict_names_column = {
    'Меторождение': 'nameReservoir',
    '№ скважины': 'wellNumberColumn',
    'Дата': 'nameDate',
    'Объекты работы': 'workHorizon',
    'Добыча нефти за посл.месяц, т': 'oilProduction',
    'Добыча жидкости за посл.месяц, т': 'fluidProduction',
    'Закачка за посл.месяц, м3': 'waterInjection',
    'Время работы в добыче, часы': 'timeProduction',
    'Время работы под закачкой, часы': 'timeInjection',
    "Координата X": 'coordinateXT1',
    "Координата Y": 'coordinateYT1',
    "Координата забоя Х (по траектории)": 'coordinateXT3',
    "Координата забоя Y (по траектории)": 'coordinateYT3',
    'Характер работы': 'workMarker',
    "Состояние": 'wellStatus'}

# CONSTANT
PROD_MARKER: str = "НЕФ"
INJ_MARKER: str = "НАГ"
# STATUS_WORK = "РАБ."
DEFAULT_HHT = 1  # meters

if __name__ == '__main__':
    # Upload files and initial data preparation_________________________________________________________________________

    data_file = "files/Вата_all.xlsx"
    name = data_file.replace("files/", "").replace(".xlsx", "")

    # upload MonthlyOperatingReport
    df_MonthlyOperatingReport = pd.read_excel(os.path.join(os.path.dirname(__file__), data_file),
                                              sheet_name="МЭР").fillna(0)
    # rename columns
    df_MonthlyOperatingReport.columns = dict_names_column.values()
    df_MonthlyOperatingReport.wellNumberColumn = df_MonthlyOperatingReport.wellNumberColumn.astype('str')
    df_MonthlyOperatingReport.workHorizon = df_MonthlyOperatingReport.workHorizon.astype('str')
    try:
        ValidatorMOR(df_dict=df_MonthlyOperatingReport.to_dict(orient="records"))
    except ValidationError as e:
        print(e)

    df_MonthlyOperatingReport = history_processing(df_MonthlyOperatingReport, PROD_MARKER, INJ_MARKER, time_work_min)

    # upload dict of effective oil height
    dict_HHT: object = pd.read_excel(os.path.join(os.path.dirname(__file__), data_file), dtype={'Скважина': str},
                                     sheet_name="Толщины").set_index('Скважина').to_dict('index')

    # upload initial_coefficient and reservoir_properties
    with open('conf_files/initial_coefficient.yml') as f:
        initial_coefficient = pd.DataFrame(yaml.safe_load(f))
    with open('conf_files/reservoir_properties.yml', 'rt', encoding='utf8') as yml:
        reservoir_properties = yaml.load(yml, Loader=yaml.Loader)

    # calculation for each reservoir:___________________________________________________________________________________
    list_reservoirs = list(df_MonthlyOperatingReport.nameReservoir.unique())

    # check for calculation with drainage zones: are there properties for all reservoirs
    if drainage_areas:
        difference = set(list_reservoirs).difference(set(list(reservoir_properties.keys())))
        if len(difference) > 0:
            raise KeyError(f"There is no properties for reservoirs: {difference}")

    # check dictionary for this reservoir
    for reservoir in list_reservoirs:
        print(f"calculate {reservoir}")
        df_MOR_reservoir = df_MonthlyOperatingReport.loc[df_MonthlyOperatingReport.nameReservoir == reservoir]
        last_data = pd.Timestamp(np.sort(df_MonthlyOperatingReport.nameDate.unique())[-1])

        # upload reservoir_properties
        if drainage_areas:
            actual_reservoir_properties = reservoir_properties.get(reservoir)

        # create DataFrame with coordinates for each well
        df_Coordinates = df_MOR_reservoir[['wellNumberColumn',
                                           'workHorizon',
                                           'workMarker',
                                           'coordinateXT1',
                                           'coordinateYT1',
                                           'coordinateXT3',
                                           'coordinateYT3',
                                           'wellStatus']].loc[df_MOR_reservoir.nameDate == last_data]
        df_Coordinates["length of well T1-3"] = np.sqrt(np.power(df_Coordinates.coordinateXT3 -
                                                                 df_Coordinates.coordinateXT1, 2)
                                                        + np.power(df_Coordinates.coordinateYT3 -
                                                                   df_Coordinates.coordinateYT1, 2))
        df_Coordinates["well type"] = 0
        df_Coordinates.loc[df_Coordinates["length of well T1-3"] < min_length_horWell, "well type"] = "vertical"
        df_Coordinates.loc[df_Coordinates["length of well T1-3"] >= min_length_horWell, "well type"] = "horizontal"
        df_Coordinates.loc[df_Coordinates["well type"] == "vertical", 'coordinateXT3'] = df_Coordinates.coordinateXT1
        df_Coordinates.loc[df_Coordinates["well type"] == "vertical", 'coordinateYT3'] = df_Coordinates.coordinateYT1

        listWellsInj = list(df_MOR_reservoir.loc[(df_MOR_reservoir.nameDate == last_data)
                                                 & (df_MOR_reservoir.workMarker == INJ_MARKER)
                                                 ].wellNumberColumn.unique())
        #  sample of Dataframe: injCelles
        df_injCells = pd.DataFrame(columns=["№ добывающей", "Ячейка", "Объект",
                                            "Дата запуска ячейки", "Расстояние, м", "Нн, м", "Нд, м",
                                            "Кдоб", "Кнаг", "Куч", "Квл",
                                            "Куч*Квл", "Куч доб", "Куч доб Итог"])

        # 0. Calculate drainage and injection zones for all wells_______________________________________________________
        df_drainage_areas = pd.DataFrame()
        actual_reservoir_properties = {}
        drainage_radius, cumulative_parameter = 0, 0
        if drainage_areas:
            df_drainage_areas = pd.DataFrame(columns=["wellNumberColumn", "dict_properties", "H_well", "type_well",
                                                      "len_well", "cumulative_parameter", "drainage_radius",
                                                      'drainage_area'])
            for well in df_Coordinates.wellNumberColumn.unique():
                # properties of horizon
                list_workHorizons = df_Coordinates.loc[df_Coordinates.wellNumberColumn == well].workHorizon.iloc[0] \
                    .replace(" ", "").split(",")
                dict_properties = get_properties(actual_reservoir_properties, list_workHorizons)
                m, So, So_min = dict_properties["m"], dict_properties["So"], dict_properties["So_min"]

                H_well = float(dict_HHT.get(well, {"HHT": DEFAULT_HHT})["HHT"])
                type_well = df_Coordinates.loc[df_Coordinates.wellNumberColumn == well]["well type"].iloc[0]
                len_well = df_Coordinates.loc[df_Coordinates.wellNumberColumn == well]["length of well T1-3"].iloc[0]

                # coordinates of well
                x_t1 = df_Coordinates.loc[df_Coordinates.wellNumberColumn == well].coordinateXT1.iloc[0]
                y_t1 = df_Coordinates.loc[df_Coordinates.wellNumberColumn == well].coordinateYT1.iloc[0]
                x_t3 = df_Coordinates.loc[df_Coordinates.wellNumberColumn == well].coordinateXT3.iloc[0]
                y_t3 = df_Coordinates.loc[df_Coordinates.wellNumberColumn == well].coordinateYT3.iloc[0]

                # find the accumulated parameter depending on the type of well
                slice_well = df_MOR_reservoir.loc[df_MOR_reservoir.wellNumberColumn == well]
                marker_well = slice_well.workMarker.iloc[-1]
                if marker_well == PROD_MARKER:
                    cumulative_parameter = slice_well.oilProduction.cumsum().iloc[-1]
                    Bo = dict_properties["Bo"]
                    Ro = dict_properties["Ro"]
                    drainage_radius = R_prod(cumulative_parameter, Bo, Ro, H_well, m, So, So_min, type_well, len_well)

                elif marker_well == INJ_MARKER:
                    cumulative_parameter = slice_well.waterInjection.cumsum().iloc[-1]
                    Bw = dict_properties["Bw"]
                    drainage_radius = R_inj(cumulative_parameter, Bw, H_well, m, So, So_min, type_well, len_well)

                drainage_area = get_polygon_well(drainage_radius, type_well, x_t1, y_t1, x_t3, y_t3)
                new_row = [well, dict_properties, H_well, type_well, len_well, cumulative_parameter, drainage_radius,
                           drainage_area]
                df_drainage_areas = df_drainage_areas.append(pd.DataFrame([new_row], index=[well],
                                                                          columns=df_drainage_areas.columns))
            """ map for all zones:
            df_drainage_areas['area'] = gpd.GeoSeries(df_drainage_areas.drainage_area).area
            df_drainage_areas = df_drainage_areas.sort_values(by=['area'], ascending=False)
            gpd.GeoSeries(df_drainage_areas.drainage_area).plot(cmap="Blues")
            plt.gca().axis("equal")
            plt.show()"""

        #  I. Start calculation of injCelle for each inj well___________________________________________________________
        for wellNumberInj in listWellsInj:
            print(f"I.{str(wellNumberInj)} "
                  f"{str(int(100 * (listWellsInj.index(wellNumberInj) + 1) / len(listWellsInj)))}%")
            slice_well = df_MOR_reservoir.loc[df_MOR_reservoir.wellNumberColumn == wellNumberInj]
            H_inj_well = float(dict_HHT.get(wellNumberInj, {"HHT": DEFAULT_HHT})["HHT"])
            df_OneInjCelle = cell_definition(slice_well, df_Coordinates, dict_HHT, df_drainage_areas,
                                             wellNumberInj, drainage_areas,
                                             max_overlap_percent=max_overlap_percent,
                                             maximum_distance=maximum_distance,
                                             angle_verWell=angle_verWell,
                                             angle_horWell_T1=angle_horWell_T1,
                                             angle_horWell_T3=angle_horWell_T3,
                                             DEFAULT_HHT=DEFAULT_HHT,
                                             PROD_MARKER=PROD_MARKER)
            df_injCells = df_injCells.append(df_OneInjCelle, ignore_index=True)

        # Sheet "Ячейки"
        df_injCells = calculation_coefficients(df_injCells, initial_coefficient)

        listWellsInj = list(df_injCells["Ячейка"].unique())

        #  II. Calculate oil increment for each injection well__________________________________________________________
        df_one_inj_well = pd.DataFrame()  # df for one inj well
        df_one_prod_well = pd.DataFrame()  # df for one prod well
        df_final_inj_well = pd.DataFrame()  # df for all inj well
        df_final_prod_well = pd.DataFrame()  # df for all prod well

        for wellNumberInj in listWellsInj:
            print(f"II.{str(wellNumberInj)} "
                  f"{str(int(100 * (listWellsInj.index(wellNumberInj) + 1) / len(listWellsInj)))}%")

            # parameters of inj well
            slice_well_inj = df_MOR_reservoir.loc[df_MOR_reservoir.wellNumberColumn ==
                                                  wellNumberInj].reset_index(drop=True)
            slice_well_inj["Injection, m3/day"] = slice_well_inj.waterInjection / \
                                                  (slice_well_inj.timeInjection / 24)
            list_wells = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)]["№ добывающей"].to_list()
            min_start_date = df_injCells["Дата запуска ячейки"].min()
            start_date_inj = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)]["Дата запуска ячейки"].iloc[0]
            #  sample of Dataframe: df_one_inj_well
            df_one_inj_well = pd.DataFrame(np.array(["Date",
                                                     'Qliq_fact, tons/day',
                                                     'Qoil_fact, tons/day',
                                                     "delta_Qliq, tons/day",
                                                     "delta_Qoil, tons/day",
                                                     "Number of working wells",
                                                     "Injection, m3/day",
                                                     "Current injection ratio, %",
                                                     "Сumulative fluid production, tons",
                                                     "Сumulative water injection, tons",
                                                     "Injection ratio, %"]), columns=['Параметр'])
            for prod_well in list_wells:
                #  sample of Dataframe: df_one_prod_well
                df_one_prod_well = pd.DataFrame(np.array(["Date",
                                                          'Qliq_fact, tons/day',
                                                          'Qoil_fact, tons/day',
                                                          "delta_Qliq, tons/day",
                                                          "delta_Qoil, tons/day",
                                                          "Сumulative fluid production, tons"]), columns=['Параметр'])
                slice_well_prod = df_MOR_reservoir.loc[df_MOR_reservoir.wellNumberColumn
                                                       == prod_well].reset_index(drop=True)

                if dynamic_coefficient:
                    name_coefficient = "Дин Куч доб Итог"
                else:
                    name_coefficient = "Куч доб Итог"
                coefficient_prod_well = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)
                                                        & (df_injCells["№ добывающей"] == prod_well
                                                           )][name_coefficient].iloc[0]

                slice_well_prod.loc[:, ("oilProduction", "fluidProduction")] = \
                    slice_well_prod.loc[:, ("oilProduction", "fluidProduction")] * coefficient_prod_well

                df_one_prod_well.insert(0, "№ добывающей", prod_well)
                df_one_prod_well.insert(0, "Ячейка", wellNumberInj)

                # add columns of date
                df_one_prod_well[pd.date_range(start=min_start_date, end=last_data, freq='MS')] = np.NAN
                df_one_prod_well.loc[0, 3:] = df_one_prod_well.columns[3:]

                # Calculate increment for each prod well________________________________________________________
                slice_well_gain = calculate_production_gain(slice_well_prod, start_date_inj)
                marker_arps = slice_well_gain[2]
                marker = slice_well_gain[1]
                slice_well_gain = slice_well_gain[0].set_index("nameDate")

                for column in slice_well_gain.columns:
                    position = list(slice_well_gain.columns).index(column) + 1
                    df_one_prod_well.iloc[position, 3:] = slice_well_gain[column] \
                        .combine_first(df_one_prod_well.iloc[position, 3:])
                    if column == "accum_liquid_fact":
                        df_one_prod_well.iloc[position, 3:] = df_one_prod_well.iloc[position, 3:].ffill(axis=0)
                df_one_prod_well = df_one_prod_well.fillna(0)

                df_one_prod_well.insert(2, "Статус", marker)
                df_one_prod_well.insert(3, "Арпс/Полка", marker_arps)
                df_final_prod_well = pd.concat([df_final_prod_well, df_one_prod_well],
                                               axis=0, sort=False).reset_index(drop=True)

            # add cell sum in df_one_inj_well
            df_one_inj_well.insert(0, "Ячейка", wellNumberInj)

            # add columns of date
            df_one_inj_well[pd.date_range(start=min_start_date, end=last_data, freq='MS')] = np.NAN
            df_one_inj_well.loc[0, 2:] = df_one_inj_well.columns[2:]

            df_one_inj_well.iloc[1, 2:] = df_final_prod_well[(df_final_prod_well["Ячейка"] == wellNumberInj) &
                                                             (df_final_prod_well["Параметр"] == 'Qliq_fact, tons/day')
                                                             ].sum(axis=0).iloc[5:]
            df_one_inj_well.iloc[2, 2:] = df_final_prod_well[(df_final_prod_well["Ячейка"] == wellNumberInj) &
                                                             (df_final_prod_well["Параметр"] == 'Qoil_fact, tons/day')
                                                             ].sum(axis=0).iloc[5:]
            df_one_inj_well.iloc[3, 2:] = df_final_prod_well[(df_final_prod_well["Ячейка"] == wellNumberInj) &
                                                             (df_final_prod_well["Параметр"] == 'delta_Qliq, tons/day')
                                                             ].sum(axis=0).iloc[5:]
            df_one_inj_well.iloc[4, 2:] = df_final_prod_well[(df_final_prod_well["Ячейка"] == wellNumberInj) &
                                                             (df_final_prod_well["Параметр"] == 'delta_Qoil, tons/day')
                                                             ].sum(axis=0).iloc[5:]
            df_one_inj_well.iloc[5, 2:] = df_final_prod_well.loc[(df_final_prod_well["Ячейка"] == wellNumberInj) &
                                                                 (df_final_prod_well[
                                                                      "Параметр"] == 'Qliq_fact, tons/day'
                                                                  )].dropna(axis=1).astype(bool).sum()[5:]
            series_injection = slice_well_inj[["Injection, m3/day",
                                               "nameDate"]].set_index("nameDate")["Injection, m3/day"]
            df_one_inj_well.iloc[6, 2:] = series_injection.combine_first(df_one_inj_well.iloc[6, 2:])

            df_one_inj_well.iloc[7, 2:] = round((df_one_inj_well.iloc[6, 2:] * volume_factor * Rw)
                                                .div(df_one_inj_well.iloc[1, 2:].where(df_one_inj_well.iloc[1, 2:] != 0,
                                                                                       np.nan)).fillna(0) * 100, 0)

            df_one_inj_well.iloc[8, 2:] = df_final_prod_well.loc[(df_final_prod_well["Ячейка"] == wellNumberInj) &
                                                                 (df_final_prod_well[
                                                                      "Параметр"] == "Сumulative fluid production, tons"
                                                                  )].sum(axis=0).iloc[5:]

            series_injection_accum = slice_well_inj[["waterInjection",
                                                     "nameDate"]].set_index("nameDate").cumsum()["waterInjection"]
            df_one_inj_well.iloc[9, 2:] = series_injection_accum.combine_first(df_one_inj_well.iloc[9, 2:]).ffill(axis=0)

            df_one_inj_well.iloc[10, 2:] = round((df_one_inj_well.iloc[9, 2:] * volume_factor * Rw)
                                                 .div(df_one_inj_well.iloc[8, 2:]
                                                      .where(df_one_inj_well.iloc[8, 2:] != 0, np.nan))
                                                 .fillna(0) * 100, 0)
            df_one_inj_well.insert(1, "тек. Комп на посл. месяц, %", df_one_inj_well.iloc[7, -1])
            df_one_inj_well.insert(1, "накоп. Комп на посл. месяц, %", df_one_inj_well.iloc[10, -1])
            df_one_inj_well = df_one_inj_well.fillna(0)
            df_final_inj_well = pd.concat([df_final_inj_well, df_one_inj_well], axis=0, sort=False) \
                .reset_index(drop=True)

        # IV. Integral effect
        """
        df_integralEffect = pd.DataFrame()
        df_forecasts = pd.DataFrame()
        for wellNumberInj in listWellsInj:
            df_part_integralEffect = pd.DataFrame(np.array([nameDate, 'Qliq_fact, т/сут', 'Qoil_fact, т/сут',
                                                            "Кол-во скв в работе, шт", "I. dQн (ХВ), т/сут",
                                                            "II. dQж (Арпс), т/сут"]), columns=['Параметр'])
            df_part_forecasts = pd.DataFrame(np.array(["Прогноз dQн, т/сут",
                                                       "Прогноз dQж, т/сут"]), columns=['Параметр'])

            slice_wellInj = df_Arpsall.loc[
                (df_Arpsall["Ячейка"] == wellNumberInj) & (df_Arpsall["№ добывающей"] == "Сумма")]
            dates_inj = slice_wellInj.loc[slice_wellInj["Параметр"] == nameDate].dropna(axis=1).iloc[0, 5:]

            df_part_integralEffect.insert(0, "Ячейка", wellNumberInj)
            df_part_forecasts.insert(0, "Последняя дата работы", dates_inj.iloc[-1])
            df_part_forecasts.insert(0, "Ячейка", wellNumberInj)

            df_part_integralEffect[list(range(dates_inj.size))] = 0
            df_part_forecasts[list(range(time_predict))] = 0

            df_part_integralEffect.iloc[0, 2:] = dates_inj
            df_part_integralEffect.iloc[1, 2:] = slice_wellInj.loc[slice_wellInj["Параметр"] ==
                                                                   'Qliq_fact, т/сут'].dropna(axis=1).iloc[0, 5:]
            df_part_integralEffect.iloc[2, 2:] = slice_wellInj.loc[slice_wellInj["Параметр"] ==
                                                                   'Qoil_fact, т/сут'].dropna(axis=1).iloc[0, 5:]
            df_part_integralEffect.iloc[5, 2:] = slice_wellInj.loc[slice_wellInj["Параметр"] ==
                                                                   "delta_Qliq, т/сут"].dropna(axis=1).iloc[0, 5:]

            slice_numWells = df_Arpsall.loc[(df_Arpsall["Ячейка"] == wellNumberInj) &
                                            (df_Arpsall["№ добывающей"] != "Сумма") &
                                            (df_Arpsall["Параметр"] == 'Qliq_fact, т/сут')].dropna(axis=1).astype(
                bool).sum()
            df_part_integralEffect.iloc[3, 2:] = slice_numWells[5:]

            slice_wellInj = df_CDall.loc[(df_CDall["Ячейка"] == wellNumberInj) &
                                         (df_CDall["№ добывающей"] == "Сумма") &
                                         (df_CDall["Параметр"] == "Прирост по ХВ, т/сут")].dropna(axis=1)
            df_part_integralEffect.iloc[4, 2:] = slice_wellInj.iloc[0, 4:]

            df_part_forecasts.iloc[0, 3:] = 1
            df_part_forecasts.iloc[1, 3:] = 1

            df_integralEffect = pd.concat([df_integralEffect, df_part_integralEffect], axis=0, sort=False).reset_index(
                drop=True)
            df_forecasts = pd.concat([df_forecasts, df_part_forecasts], axis=0, sort=False).reset_index(
                drop=True)
        print(1)"""

        df_injCells["Ячейка"] = df_injCells["Ячейка"].astype("str")
        df_injCells = df_injCells.sort_values(by="Ячейка").reset_index(drop=True)
        dict_df = {f"Ячейки_{reservoir}": df_injCells, f"Прирост доб_{reservoir}": df_final_prod_well,
                   f"Прирост наг_{reservoir}": df_final_inj_well}

        # Start print in Excel
        app1 = xw.App(visible=False)
        new_wb = xw.Book()

        for key in dict_df.keys():
            if f"{key}" in new_wb.sheets:
                xw.Sheet[f"{key}"].delete()
            new_wb.sheets.add(f"{key}")
            sht = new_wb.sheets(f"{key}")
            sht.range('A1').options().value = dict_df[key]

    new_wb.save(str(os.path.basename(data_file)).replace(".xlsx", "") + "_out.xlsx")
    app1.kill()
    # End print
    pass
