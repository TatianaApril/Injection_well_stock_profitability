import pandas as pd
import os
import numpy as np
import geopandas as gdp
import xlwings as xw
from shapely import Point, LineString
import yaml
from pydantic import ValidationError

from FirstRowWells import first_row_of_well_geometry, first_row_of_well_drainage_front
from Utility_function_inj import calculation_coefficients, history_processing
from CharDesatur import characteristic_of_desaturation
from Arps_functions import GainCell_Arps
from Schema import ValidatorMOR
from drainage_area import R_inj, R_prod, get_properties

# Parameters
maximum_distance: int = 1000  # maximum distance from injection well for reacting wells
min_length_horWell = 150  # minimum length between points T1 and T3 to consider the well as horizontal
max_overlap_percent = 20  # how much one well can cover the sector of another for selection in the first row of wells
angle_verWell = 15  # degree: sector for vertical wells
angle_horWell_T1 = angle_verWell  # sector expansion angle for horizontal well's T1 point
angle_horWell_T3 = 5  # sector expansion angle for horizontal well's T3 point
time_predict = 36  # the number of months in the forecast

# Switches
drainage_areas = True

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
PROD_MARKER = "НЕФ"
INJ_MARKER = "НАГ"
STATUS_WORK = "РАБ."
DEFAULT_HHT = 1  # meters

if __name__ == '__main__':
    # Upload files and initial data preparation_________________________________________________________________________

    data_file = "files/Копия Аспид ппд2.xlsx"
    name = data_file.replace("files/", "").replace(".xlsx", "")

    # upload MonthlyOperatingReport
    df_MonthlyOperatingReport = history_processing(pd.read_excel(os.path.join(os.path.dirname(__file__), data_file),
                                                                 sheet_name="МЭР").fillna(0))
    # rename columns
    df_MonthlyOperatingReport.columns = dict_names_column.values()
    df_MonthlyOperatingReport.wellNumberColumn = df_MonthlyOperatingReport.wellNumberColumn.astype('str')
    df_MonthlyOperatingReport.workHorizon = df_MonthlyOperatingReport.workHorizon.astype('str')
    try:
        ValidatorMOR(df_dict=df_MonthlyOperatingReport.to_dict(orient="records"))
    except ValidationError as e:
        print(e)

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
        if len(difference):
            raise KeyError(f"There is no properties for reservoirs: {difference}")

    # check dictionary for this reservoir
    for reservoir in list_reservoirs:

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

        #  I. Start calculation of injCelle for each inj well___________________________________________________________
        for wellNumberInj in listWellsInj:
            df_OneInjCelle = pd.DataFrame()
            print(f"I.{str(wellNumberInj)} "
                  f"{str(int(100 * (listWellsInj.index(wellNumberInj) + 1) / len(listWellsInj)))}%")
            # parameters of inj well
            type_inj_well = df_Coordinates.loc[df_Coordinates.wellNumberColumn == wellNumberInj]["well type"].iloc[0]
            len_inj_well = df_Coordinates.loc[df_Coordinates.wellNumberColumn
                                              == wellNumberInj]["length of well T1-3"].iloc[0]
            workHorizonInj = df_Coordinates.loc[df_Coordinates.wellNumberColumn == wellNumberInj].workHorizon.iloc[0]
            list_workHorizonInj = workHorizonInj.split(",")
            H_inj_well = float(dict_HHT.get(wellNumberInj, {"HHT": DEFAULT_HHT})["HHT"])

            # wells producing one layer == injection well horizon
            df_WellOneHorizon = df_Coordinates
            df_WellOneHorizon = df_WellOneHorizon[
                list(map(lambda x: len(set(x.split(",")) & set(list_workHorizonInj)) > 0,
                         df_WellOneHorizon.workHorizon))]
            if df_WellOneHorizon.shape[0] > 1:
                df_WellOneHorizon = df_WellOneHorizon[['wellNumberColumn',
                                                       'workMarker',
                                                       'coordinateXT1',
                                                       'coordinateYT1',
                                                       'coordinateXT3',
                                                       'coordinateYT3',
                                                       "well type",
                                                       "length of well T1-3"]].set_index("wellNumberColumn")

                # add shapely types for well coordinates
                df_WellOneHorizon["POINT T1"] = list(map(lambda x, y: Point(x, y),
                                                         df_WellOneHorizon.coordinateXT1,
                                                         df_WellOneHorizon.coordinateYT1))
                df_WellOneHorizon["POINT T3"] = list(map(lambda x, y: Point(x, y),
                                                         df_WellOneHorizon.coordinateXT3,
                                                         df_WellOneHorizon.coordinateYT3))
                df_WellOneHorizon["LINESTRING"] = list(map(lambda x, y: LineString([x, y]),
                                                           df_WellOneHorizon["POINT T1"],
                                                           df_WellOneHorizon["POINT T3"]))

                gdf_WellOneHorizon = gdp.GeoDataFrame(df_WellOneHorizon, geometry="LINESTRING")
                line_inj = gdf_WellOneHorizon["LINESTRING"].loc[wellNumberInj]
                gdf_WellOneHorizon['distance'] = gdf_WellOneHorizon["LINESTRING"].distance(line_inj)

                if drainage_areas:
                    # check properties of inj well
                    Bo, Bw, Ro, m, So, So_min = get_properties(actual_reservoir_properties, list_workHorizonInj)
                    cumulative_water_inj = df_MOR_reservoir.loc[df_MOR_reservoir.wellNumberColumn
                                                                == wellNumberInj].cumsum()
                    injection_radius = R_inj(cumulative_water_inj, Bw, H_inj_well, m, So, So_min, type_inj_well,
                                             len_inj_well)
                    "______________!!!!!!!!!!!!!!!!!!!!!!!!!________________________"
                    # select wells in the injection zone
                    gdf_WellOneArea = gdf_WellOneHorizon[(gdf_WellOneHorizon['distance'] < maximum_distance)]
                    df_WellOneArea = pd.DataFrame(gdf_WellOneArea)
                    # select first row of wells based on drainage areas
                    list_first_row_wells = first_row_of_well_drainage_front(gdf_WellOneArea, wellNumberInj)
                else:
                    # select wells in the injection zone (distance < maximumDistance)
                    gdf_WellOneArea = gdf_WellOneHorizon[(gdf_WellOneHorizon['distance'] < maximum_distance)]
                    df_WellOneArea = pd.DataFrame(gdf_WellOneArea)
                    # select first row of wells based on geometry
                    list_first_row_wells = first_row_of_well_geometry(df_WellOneArea, wellNumberInj,
                                                                      angle_verWell, max_overlap_percent,
                                                                      angle_horWell_T1, angle_horWell_T3)

                df_WellOneArea = df_WellOneArea[df_WellOneArea.index.isin(list_first_row_wells)]
                df_WellOneArea = df_WellOneArea.loc[df_WellOneArea.workMarker == PROD_MARKER]
                df_OneInjCelle["№ добывающей"] = df_WellOneArea.index
                df_OneInjCelle["Ячейка"] = wellNumberInj
                df_OneInjCelle["Объект"] = workHorizonInj
                df_work_time = df_MOR_reservoir.loc[(df_MOR_reservoir.wellNumberColumn == wellNumberInj) &
                                                    (df_MOR_reservoir.wellStatus == STATUS_WORK) &
                                                    (df_MOR_reservoir.workHorizon == workHorizonInj)] \
                    .nameDate
                if df_work_time.empty:
                    print("нет рабочих дней")
                    continue
                start_date_inj = df_work_time.iloc[0]
                df_OneInjCelle["Дата запуска ячейки"] = start_date_inj
                df_OneInjCelle["Расстояние, м"] = df_WellOneArea['distance'].values
                df_OneInjCelle["Нн, м"] = H_inj_well
                if df_OneInjCelle.empty:
                    print("нет окружения")
                df_OneInjCelle["Нд, м"] = df_OneInjCelle["№ добывающей"].apply(
                    lambda x: float(dict_HHT.get(x, {"HHT": DEFAULT_HHT})["HHT"]))
                df_OneInjCelle[["Кдоб", "Кнаг", "Куч", "Квл", "Куч*Квл", "Куч доб", "Куч доб Итог"]] = 0
                df_injCells = df_injCells.append(df_OneInjCelle, ignore_index=True)
            else:
                print("нет окружения")
                exit()

        # Sheet "Ячейки"
        df_injCells = calculation_coefficients(df_injCells, initial_coefficient)

        listWellsInj = list(df_injCells["Ячейка"].unique())

        #  II. Fist part of production gain  - characteristic_of_desaturation
        df_CDCells = pd.DataFrame()  # df for one inj well
        df_part_CD = pd.DataFrame()  # df for one prod well
        df_CDall = pd.DataFrame()  # df for all inj well

        #  III. Second part of production gain  - Arps of fluid flow
        df_ArpsCells = pd.DataFrame()  # df for one inj well
        df_part_Arps = pd.DataFrame()  # df for one prod well
        df_Arpsall = pd.DataFrame()  # df for all inj well

        for wellNumberInj in listWellsInj:
            print(
                f"II.{str(wellNumberInj)} {str(int(100 * (listWellsInj.index(wellNumberInj) + 1) / len(listWellsInj)))}%")
            slice_wellInj = df_MOR_reservoir.loc[df_MOR_reservoir[wellNumberColumn]
                                                 == wellNumberInj]
            listWells_Cell = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)]["№ добывающей"].to_list()
            start_date_inj = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)]["Дата запуска ячейки"].iloc[0]

            df_part_Arps = pd.DataFrame(np.array([nameDate, 'Qliq_fact, т/сут', 'Qoil_fact, т/сут',
                                                  "Обводненность, д. ед.", "Qliq_before_inj, т/сут",
                                                  "delta_Qliq, т/сут"]), columns=['Параметр'])
            df_part_CD = pd.DataFrame(np.array([nameDate, oilProduction, fluidProduction, "Прирост по ХВ, т",
                                                "Прирост по ХВ, т/сут"]), columns=['Параметр'])
            for prodWell_inCell in listWells_Cell:
                slice_wellProd = df_MOR_reservoir.loc[df_MOR_reservoir[wellNumberColumn]
                                         == prodWell_inCell]
                if slice_wellProd.empty:
                    continue
                coeff_prodWell = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)
                                                 & (df_injCells["№ добывающей"] == prodWell_inCell)][
                    "Куч доб Итог"].iloc[
                    0]
                slice_wellProd[[oilProduction, fluidProduction]] = slice_wellProd[
                                                                       [oilProduction,
                                                                        fluidProduction]] * coeff_prodWell
                slice_wellProd["Прирост по ХВ, т"] = 0
                slice_wellProd["Прирост по ХВ, т/сут"] = 0

                df_part_CD.insert(0, "№ добывающей", prodWell_inCell)
                df_part_CD.insert(0, "Ячейка", wellNumberInj)
                df_part_CD[slice_wellInj.loc[slice_wellInj[nameDate] >= start_date_inj][nameDate]] = 0
                df_part_CD.iloc[0, 3:] = slice_wellInj.loc[slice_wellInj[nameDate] >= start_date_inj][nameDate]

                df_part_Arps.insert(0, "№ добывающей", prodWell_inCell)
                df_part_Arps.insert(0, "Ячейка", wellNumberInj)
                df_part_Arps[slice_wellInj.loc[slice_wellInj[nameDate] >= start_date_inj][nameDate]] = 0
                df_part_Arps.iloc[0, 3:] = slice_wellInj.loc[slice_wellInj[nameDate] >= start_date_inj][nameDate]

                slice_well_Arps = pd.DataFrame()
                if slice_wellProd.loc[slice_wellProd[nameDate] < start_date_inj].empty:
                    marker = 'запущена после ППД'
                else:
                    size = slice_wellProd.loc[slice_wellProd[nameDate] < start_date_inj].shape[0]
                    if size <= 3:
                        marker = 'меньше 4х месяцев работы без ппд'
                    else:
                        marker = f'до ППД отработала {str(size)} месяцев'
                        index_start = slice_wellProd[slice_wellProd[nameDate] <= start_date_inj].index.tolist()[-1]
                        if index_start in slice_wellProd.index.tolist()[-3:]:
                            marker = f'Нет истории работы после запуска ППД'
                        else:
                            slice_well_gainСD = characteristic_of_desaturation(slice_wellProd, start_date_inj, nameDate,
                                                                               oilProduction, fluidProduction,
                                                                               timeProduction)
                            slice_well_gainСD = slice_well_gainСD.set_index(nameDate)

                            df_part_CD.iloc[1, 3:] = slice_well_gainСD[oilProduction][size:].combine_first(
                                df_part_CD.iloc[1, 3:])
                            df_part_CD.iloc[2, 3:] = slice_well_gainСD[fluidProduction][size:].combine_first(
                                df_part_CD.iloc[2, 3:])
                            df_part_CD.iloc[3, 3:] = slice_well_gainСD["Прирост по ХВ, т"][size:].combine_first(
                                df_part_CD.iloc[3, 3:])
                            df_part_CD.iloc[4, 3:] = slice_well_gainСD["Прирост по ХВ, т/сут"][size:].combine_first(
                                df_part_CD.iloc[4, 3:])

                            slice_well_Arps = GainCell_Arps(slice_wellProd, start_date_inj, oilProduction,
                                                            fluidProduction,
                                                            timeProduction, nameDate)
                            slice_well_Arps = slice_well_Arps.set_index(nameDate)

                            df_part_Arps.iloc[1, 3:] = slice_well_Arps['Qliq_fact, т/сут'].combine_first(
                                df_part_Arps.iloc[1, 3:])
                            df_part_Arps.iloc[2, 3:] = slice_well_Arps['Qoil_fact, т/сут'].combine_first(
                                df_part_Arps.iloc[2, 3:])
                            df_part_Arps.iloc[3, 3:] = slice_well_Arps["Обводненность, д. ед."].combine_first(
                                df_part_Arps.iloc[3, 3:])
                            df_part_Arps.iloc[4, 3:] = slice_well_Arps["Qliq_before_inj, т/сут"].combine_first(
                                df_part_Arps.iloc[4, 3:])
                            df_part_Arps.iloc[5, 3:] = slice_well_Arps["delta_Qliq, т/сут"].combine_first(
                                df_part_Arps.iloc[5, 3:])

                df_part_CD.insert(2, "Статус", marker)
                df_part_Arps.insert(2, "Статус", marker)
                if not slice_well_Arps.empty:
                    df_part_Arps.insert(3, "Арпс/Полка", slice_well_Arps["Арпс/Полка"].unique()[0])
                else:
                    df_part_Arps.insert(3, "Арпс/Полка", "нет расчета")

                df_CDCells = pd.concat([df_CDCells, df_part_CD], axis=0, sort=False).reset_index(drop=True)
                df_ArpsCells = pd.concat([df_ArpsCells, df_part_Arps], axis=0, sort=False).reset_index(drop=True)

                df_part_CD = pd.DataFrame(np.array([nameDate, oilProduction, fluidProduction, "Прирост по ХВ, т",
                                                    "Прирост по ХВ, т/сут"]), columns=['Параметр'])
                df_part_Arps = pd.DataFrame(np.array([nameDate, 'Qliq_fact, т/сут', 'Qoil_fact, т/сут',
                                                      "Обводненность, д. ед.", "Qliq_before_inj, т/сут",
                                                      "delta_Qliq, т/сут"]), columns=['Параметр'])

            df_part_CD.insert(0, "Статус", "ППД")
            df_part_CD.insert(0, "№ добывающей", "Сумма")
            df_part_CD.insert(0, "Ячейка", wellNumberInj)

            df_part_CD[slice_wellInj.loc[slice_wellInj[nameDate] >= start_date_inj][nameDate]] = 0
            df_part_CD.iloc[0, 4:] = slice_wellInj.loc[slice_wellInj[nameDate] >= start_date_inj][nameDate]

            df_part_CD.iloc[1, 4:] = df_CDCells[(df_CDCells["Ячейка"] == wellNumberInj) &
                                                (df_CDCells["Параметр"] == oilProduction)].sum(axis=0).iloc[4:]
            df_part_CD.iloc[2, 4:] = df_CDCells[(df_CDCells["Ячейка"] == wellNumberInj) &
                                                (df_CDCells["Параметр"] == fluidProduction)].sum(axis=0).iloc[4:]
            df_part_CD.iloc[4, 4:] = df_CDCells[(df_CDCells["Ячейка"] == wellNumberInj) &
                                                (df_CDCells["Параметр"] == "Прирост по ХВ, т/сут")].sum(axis=0).iloc[4:]
            df_part_CD.iloc[3, 4:] = df_CDCells[(df_CDCells["Ячейка"] == wellNumberInj) &
                                                (df_CDCells["Параметр"] == "Прирост по ХВ, т")].sum(axis=0).iloc[4:]

            df_CDCells = pd.concat([df_CDCells, df_part_CD], axis=0, sort=False).reset_index(drop=True)
            df_CDCells.columns = list(df_CDCells.columns)[:4] + list(range(df_CDCells.shape[1] - 4))
            df_CDall = pd.concat([df_CDall, df_CDCells], axis=0, sort=False).reset_index(drop=True)
            df_CDCells = pd.DataFrame()

            df_part_Arps.insert(0, "Арпс/Полка", "Сумма")
            df_part_Arps.insert(0, "Статус", "ППД")
            df_part_Arps.insert(0, "№ добывающей", "Сумма")
            df_part_Arps.insert(0, "Ячейка", wellNumberInj)

            df_part_Arps[slice_wellInj.loc[slice_wellInj[nameDate] >= start_date_inj][nameDate]] = 0
            df_part_Arps.iloc[0, 5:] = slice_wellInj.loc[slice_wellInj[nameDate] >= start_date_inj][nameDate]

            df_part_Arps.iloc[1, 5:] = df_ArpsCells[(df_ArpsCells["Ячейка"] == wellNumberInj) &
                                                    (df_ArpsCells["Параметр"] == 'Qliq_fact, т/сут')].sum(axis=0).iloc[
                                       5:]
            df_part_Arps.iloc[2, 5:] = df_ArpsCells[(df_ArpsCells["Ячейка"] == wellNumberInj) &
                                                    (df_ArpsCells["Параметр"] == 'Qoil_fact, т/сут')].sum(axis=0).iloc[
                                       5:]

            df_part_Arps.iloc[3, 5:] = 1 - df_part_Arps.iloc[2, 5:].astype('float64') / df_part_Arps.iloc[1, 5:].astype(
                'float64')
            df_part_Arps.fillna(0)
            df_part_Arps.iloc[4, 5:] = df_ArpsCells[(df_ArpsCells["Ячейка"] == wellNumberInj) &
                                                    (df_ArpsCells["Параметр"] == "Qliq_before_inj, т/сут")].sum(
                axis=0).iloc[5:]
            df_part_Arps.iloc[5, 5:] = df_ArpsCells[(df_ArpsCells["Ячейка"] == wellNumberInj) &
                                                    (df_ArpsCells["Параметр"] == "delta_Qliq, т/сут")].sum(axis=0).iloc[
                                       5:]

            df_ArpsCells = pd.concat([df_ArpsCells, df_part_Arps], axis=0, sort=False).reset_index(drop=True)
            df_ArpsCells.columns = list(df_ArpsCells.columns)[:5] + list(range(df_ArpsCells.shape[1] - 5))
            df_Arpsall = pd.concat([df_Arpsall, df_ArpsCells], axis=0, sort=False).reset_index(drop=True)
            df_ArpsCells = pd.DataFrame()

        # IV. Integral effect
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
        print(1)

        # Start print in Excel
        app1 = xw.App(visible=False)
        new_wb = xw.Book()

        if f"Ячейки_{reservoir}" in new_wb.sheets:
            xw.Sheet[f"Ячейки_{reservoir}"].delete()
        new_wb.sheets.add(f"Ячейки_{reservoir}")
        sht = new_wb.sheets(f"Ячейки_{reservoir}")
        df_injCells["Ячейка"] = df_injCells["Ячейка"].astype("str")
        df_injCells = df_injCells.sort_values(by="Ячейка").reset_index(drop=True)
        sht.range('A1').options().value = df_injCells

        if f"Прирост по ХВ_{reservoir}" in new_wb.sheets:
            xw.Sheet[f"Прирост по ХВ_{reservoir}"].delete()
        new_wb.sheets.add(f"Прирост по ХВ_{reservoir}")
        sht = new_wb.sheets(f"Прирост по ХВ_{reservoir}")
        sht.range('A1').options().value = df_CDall

        if f"Прирост Qж_{reservoir}" in new_wb.sheets:
            xw.Sheet[f"Прирост Qж_{reservoir}"].delete()
        new_wb.sheets.add(f"Прирост Qж_{reservoir}")
        sht = new_wb.sheets(f"Прирост Qж_{reservoir}")
        sht.range('A1').options().value = df_Arpsall

        if f"Прирост интегральный_{reservoir}" in new_wb.sheets:
            xw.Sheet[f"Прирост интегральный_{reservoir}"].delete()
        new_wb.sheets.add(f"Прирост интегральный_{reservoir}")
        sht = new_wb.sheets(f"Прирост интегральный_{reservoir}")
        sht.range('A1').options().value = df_integralEffect

        if f"Прогноз_{reservoir}" in new_wb.sheets:
            xw.Sheet[f"Прогноз_{reservoir}"].delete()
        new_wb.sheets.add(f"Прогноз_{reservoir}")
        sht = new_wb.sheets(f"Прогноз_{reservoir}")
        sht.range('A1').options().value = df_forecasts

        if f"Координаты_{reservoir}" in new_wb.sheets:
            xw.Sheet[f"Координаты_{reservoir}"].delete()
        new_wb.sheets.add(f"Координаты_{reservoir}")
        sht = new_wb.sheets(f"Координаты_{reservoir}")
        sht.range('A1').options().value = df_Coordinates

    new_wb.save(str(os.path.basename(data_file)).replace(".xlsx", "") + "_out.xlsx")
    app1.kill()
    # End print
    pass
