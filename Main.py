import pandas as pd
import os
import numpy as np
import geopandas as gdp
import xlwings as xw

from FirstRowWells import firstRow_of_Wells_geometry, firstRow_of_Wells_drainageFront
from Utility_function_inj import calculation_coefficients, history_processing
from CharDesatur import characteristic_of_desaturation
from Arps_functions import GainCell_Arps

# Parameters
maximumDistance = 1000  # maximum distance from injection well for reacting wells
MinLengthHorizontalWell = 150  # minimum length between points T1 and T3 to consider the well as horizontal
MaxOverlapPercent = 30  # how much one well can cover the sector of another for selection in the first row of wells
verticalWellAngle = 10  # degree
time_forecast = 36  # the number of months in the forecast

nameReservoir = 'Меторождение'
wellNumberColumn = '№ скважины'
nameDate = 'Дата'
workHorizon = 'Объекты работы'
oilProduction = 'Добыча нефти за посл.месяц, т'
fluidProduction = 'Добыча жидкости за посл.месяц, т'
waterInjection = 'Закачка за посл.месяц, м3'
timeProduction = 'Время работы в добыче, часы'
timeInjection = 'Время работы под закачкой, часы'
coordinateXT1 = "Координата X"
coordinateYT1 = "Координата Y"
coordinateXT3 = "Координата забоя Х (по траектории)"
coordinateYT3 = "Координата забоя Y (по траектории)"
workMarker = 'Характер работы'
wellStatus = "Состояние"

prodMarker = "НЕФ"
injMarker = "НАГ"
wellStatus_work = "РАБ."

listNamesColumns = [nameReservoir,
                    wellNumberColumn,
                    nameDate,
                    workHorizon,
                    oilProduction,
                    fluidProduction,
                    waterInjection,
                    timeProduction,
                    timeInjection,
                    coordinateXT1,
                    coordinateYT1,
                    coordinateXT3,
                    coordinateYT3,
                    workMarker,
                    wellStatus,
                    prodMarker,
                    injMarker,
                    wellStatus_work]

if __name__ == '__main__':
    data_file = "files/Тайлаковское.xlsx"
    #  data_file = "files/Копия Аспид ппд2.xlsx"

    # upload all sheets from book
    df_MonthlyOperatingReport = pd.read_excel(os.path.join(os.path.dirname(__file__), data_file), sheet_name="МЭР",
                                              dtype={'№ скважины': str}).fillna(0)
    dict_HHT = pd.read_excel(os.path.join(os.path.dirname(__file__), data_file), sheet_name="Толщины",
                             dtype={'Скважина': str}).set_index('Скважина').to_dict('index')
    df_coeff = pd.read_excel(os.path.join(os.path.dirname(__file__), data_file), sheet_name="Куч")

    # initial data preparation
    last_data = pd.Timestamp(np.sort(df_MonthlyOperatingReport[nameDate].unique())[-1])
    listWellsInj = list(df_MonthlyOperatingReport.loc[(df_MonthlyOperatingReport[nameDate] == last_data)
                                                      & (df_MonthlyOperatingReport[workMarker] == injMarker)
                                                      ][wellNumberColumn].unique())
    df_Coordinates = df_MonthlyOperatingReport[[wellNumberColumn,
                                                workHorizon, workMarker,
                                                coordinateXT1,
                                                coordinateYT1,
                                                coordinateXT3,
                                                coordinateYT3,
                                                wellStatus]].loc[df_MonthlyOperatingReport[nameDate] == last_data]
    df_Coordinates["length of well T1-3"] = np.sqrt(np.power(df_Coordinates[coordinateXT3] -
                                                             df_Coordinates[coordinateXT1], 2)
                                                    + np.power(df_Coordinates[coordinateYT3] -
                                                               df_Coordinates[coordinateYT1], 2))
    df_Coordinates["well type"] = 0
    df_Coordinates.loc[df_Coordinates["length of well T1-3"] < MinLengthHorizontalWell, "well type"] = "vertical"
    df_Coordinates.loc[df_Coordinates["length of well T1-3"] >= MinLengthHorizontalWell, "well type"] = "horizontal"
    df_Coordinates.loc[df_Coordinates["well type"] == "vertical", coordinateXT3] = df_Coordinates[coordinateXT1]
    df_Coordinates.loc[df_Coordinates["well type"] == "vertical", coordinateYT3] = df_Coordinates[coordinateYT1]

    #  sample of Dataframe: injCelles
    df_injCells = pd.DataFrame(columns=["№ добывающей", "Ячейка", "Объект",
                                         "Дата запуска ячейки", "Расстояние, м", "Нн, м", "Нд, м",
                                         "Кдоб", "Кнаг", "Куч", "Квл",
                                         "Куч*Квл", "Куч доб", "Куч доб Итог"])
    #  I. Start calculation of injCelle for each well
    for wellNumberInj in listWellsInj:
        df_OneInjCelle = pd.DataFrame()
        print(f"I.{str(wellNumberInj)} {str(int(100 * (listWellsInj.index(wellNumberInj) + 1) / len(listWellsInj)))}%")
        workHorizonInj = df_Coordinates.loc[df_Coordinates[wellNumberColumn] == wellNumberInj][workHorizon].iloc[0]

        df_WellOneHorizon = df_Coordinates
        df_WellOneHorizon["horizon"] = df_WellOneHorizon[workHorizon].str.find(workHorizonInj)
        df_WellOneHorizon = df_WellOneHorizon.dropna(subset=["horizon"])
        del df_WellOneHorizon["horizon"]
        if df_WellOneHorizon.shape[0] > 1:
            df_WellOneHorizon = df_WellOneHorizon[[wellNumberColumn, workMarker,
                                                   coordinateXT1,
                                                   coordinateYT1,
                                                   coordinateXT3,
                                                   coordinateYT3, "well type"]].set_index(wellNumberColumn)

            df_WellOneHorizon["LINESTRING"] = "LINESTRING(" + df_WellOneHorizon[coordinateXT1].astype(str) + " " + \
                                              df_WellOneHorizon[coordinateYT1].astype(str) + "," + \
                                              df_WellOneHorizon[coordinateXT3].astype(str) + " " + \
                                              df_WellOneHorizon[coordinateYT3].astype(str) + ")"

            gs = gdp.GeoSeries.from_wkt(df_WellOneHorizon["LINESTRING"])
            gdf_WellOneHorizon = gdp.GeoDataFrame(df_WellOneHorizon, geometry=gs)
            line_inj = gdf_WellOneHorizon['geometry'].loc[wellNumberInj]
            gdf_WellOneHorizon['distance'] = gdf_WellOneHorizon['geometry'].distance(line_inj)

            #  select wells in the injection zone
            gdf_WellOneArea = gdf_WellOneHorizon[(gdf_WellOneHorizon['distance'] < maximumDistance)]
            df_WellOneArea = pd.DataFrame(gdf_WellOneArea)
            listNamesFisrtRowWells = firstRow_of_Wells_geometry(df_WellOneArea, wellNumberInj,
                                                                coordinateXT1, coordinateYT1,
                                                                coordinateXT3, coordinateYT3,
                                                                verticalWellAngle, MaxOverlapPercent)
            """ Аналогичная функция по выделению ряда скважин но по областям дренирования/закачки - в разработке
                        listNamesFisrtRowWells = firstRow_of_Wells_drainageFront(gdf_WellOneArea, wellNumberInj,
                                                                                 coordinateXT1,
                                                                                 coordinateYT1,
                                                                                 coordinateXT3,
                                                                                 coordinateYT3, maximumDistance)"""
            df_WellOneArea = df_WellOneArea[df_WellOneArea.index.isin(listNamesFisrtRowWells)]
            df_WellOneArea = df_WellOneArea.loc[df_WellOneArea[workMarker] == prodMarker]
            df_OneInjCelle["№ добывающей"] = df_WellOneArea.index
            df_OneInjCelle["Ячейка"] = wellNumberInj
            df_OneInjCelle["Объект"] = workHorizonInj
            df_worktime = df_MonthlyOperatingReport.loc[(df_MonthlyOperatingReport[wellNumberColumn] == wellNumberInj)
                                              & (df_MonthlyOperatingReport[wellStatus] == wellStatus_work)
                                              & (df_MonthlyOperatingReport[workHorizon] == workHorizonInj)][
                    nameDate]
            if df_worktime.empty:
                print("нет рабочих дней")
                continue
            start_date_inj = df_worktime.iloc[0]
            df_OneInjCelle["Дата запуска ячейки"] = start_date_inj
            df_OneInjCelle["Расстояние, м"] = df_WellOneArea['distance'].values
            df_OneInjCelle["Нн, м"] = float(dict_HHT[wellNumberInj][list(dict_HHT[wellNumberInj].keys())[2]])
            if df_OneInjCelle.empty:
                print("нет окружения")
            try:
                df_OneInjCelle["Нд, м"] = df_OneInjCelle["№ добывающей"].apply(lambda x:
                                                                               float(dict_HHT[x][
                                                                                         list(dict_HHT[x].keys())[2]]))
            except KeyError as e:
                print(f'KeyError - reason: value of HHT well {str(e)}')
                exit()
            except:
                print('problem with dict_HHT (not KeyError)')
                exit()
            df_OneInjCelle[["Кдоб", "Кнаг", "Куч", "Квл", "Куч*Квл", "Куч доб", "Куч доб Итог"]] = 0
            df_injCells = df_injCells.append(df_OneInjCelle, ignore_index=True)
        else:
            print("нет окружения")
            exit()

    # Sheet "Ячейки"
    df_injCells = calculation_coefficients(df_injCells, df_coeff)

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
        print(f"II.{str(wellNumberInj)} {str(int(100 * (listWellsInj.index(wellNumberInj) + 1) / len(listWellsInj)))}%")
        slice_wellInj = df_MonthlyOperatingReport.loc[df_MonthlyOperatingReport[wellNumberColumn]
                                                      == wellNumberInj]
        listWells_Cell = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)]["№ добывающей"].to_list()
        start_date_inj = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)]["Дата запуска ячейки"].iloc[0]

        df_part_Arps = pd.DataFrame(np.array([nameDate, 'Qliq_fact, т/сут', 'Qoil_fact, т/сут',
                                              "Обводненность, д. ед.", "Qliq_before_inj, т/сут",
                                              "delta_Qliq, т/сут"]), columns=['Параметр'])
        df_part_CD = pd.DataFrame(np.array([nameDate, oilProduction, fluidProduction, "Прирост по ХВ, т",
                                            "Прирост по ХВ, т/сут"]), columns=['Параметр'])
        for prodWell_inCell in listWells_Cell:
            slice_wellProd = history_processing(
                df_MonthlyOperatingReport.loc[df_MonthlyOperatingReport[wellNumberColumn]
                                              == prodWell_inCell])
            if slice_wellProd.empty:
                continue
            coeff_prodWell = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)
                                             & (df_injCells["№ добывающей"] == prodWell_inCell)]["Куч доб Итог"].iloc[
                0]
            slice_wellProd[[oilProduction, fluidProduction]] = slice_wellProd[
                                                                   [oilProduction, fluidProduction]] * coeff_prodWell
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
                                                                           oilProduction, fluidProduction, timeProduction)
                        slice_well_gainСD = slice_well_gainСD.set_index(nameDate)

                        df_part_CD.iloc[1, 3:] = slice_well_gainСD[oilProduction][size:].combine_first(
                            df_part_CD.iloc[1, 3:])
                        df_part_CD.iloc[2, 3:] = slice_well_gainСD[fluidProduction][size:].combine_first(
                            df_part_CD.iloc[2, 3:])
                        df_part_CD.iloc[3, 3:] = slice_well_gainСD["Прирост по ХВ, т"][size:].combine_first(
                            df_part_CD.iloc[3, 3:])
                        df_part_CD.iloc[4, 3:] = slice_well_gainСD["Прирост по ХВ, т/сут"][size:].combine_first(
                            df_part_CD.iloc[4, 3:])

                        slice_well_Arps = GainCell_Arps(slice_wellProd, start_date_inj, oilProduction, fluidProduction,
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
                                                (df_ArpsCells["Параметр"] == 'Qliq_fact, т/сут')].sum(axis=0).iloc[5:]
        df_part_Arps.iloc[2, 5:] = df_ArpsCells[(df_ArpsCells["Ячейка"] == wellNumberInj) &
                                                (df_ArpsCells["Параметр"] == 'Qoil_fact, т/сут')].sum(axis=0).iloc[5:]

        df_part_Arps.iloc[3, 5:] = 1 - df_part_Arps.iloc[2, 5:].astype('float64') / df_part_Arps.iloc[1, 5:].astype(
            'float64')
        df_part_Arps.fillna(0)
        df_part_Arps.iloc[4, 5:] = df_ArpsCells[(df_ArpsCells["Ячейка"] == wellNumberInj) &
                                                (df_ArpsCells["Параметр"] == "Qliq_before_inj, т/сут")].sum(
            axis=0).iloc[5:]
        df_part_Arps.iloc[5, 5:] = df_ArpsCells[(df_ArpsCells["Ячейка"] == wellNumberInj) &
                                                (df_ArpsCells["Параметр"] == "delta_Qliq, т/сут")].sum(axis=0).iloc[5:]

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
        df_part_forecasts[list(range(time_forecast))] = 0

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

    if "Ячейки" in new_wb.sheets:
        xw.Sheet["Ячейки"].delete()
    new_wb.sheets.add("Ячейки")
    sht = new_wb.sheets("Ячейки")
    df_injCells["Ячейка"] = df_injCells["Ячейка"].astype("str")
    df_injCells = df_injCells.sort_values(by="Ячейка").reset_index(drop=True)
    sht.range('A1').options().value = df_injCells

    if "Прирост по ХВ" in new_wb.sheets:
        xw.Sheet["Прирост по ХВ"].delete()
    new_wb.sheets.add("Прирост по ХВ")
    sht = new_wb.sheets("Прирост по ХВ")
    sht.range('A1').options().value = df_CDall

    if "Прирост Qж" in new_wb.sheets:
        xw.Sheet["Прирост Qж"].delete()
    new_wb.sheets.add("Прирост Qж")
    sht = new_wb.sheets("Прирост Qж")
    sht.range('A1').options().value = df_Arpsall

    if "Прирост интегральный" in new_wb.sheets:
        xw.Sheet["Прирост интегральный"].delete()
    new_wb.sheets.add("Прирост интегральный")
    sht = new_wb.sheets("Прирост интегральный")
    sht.range('A1').options().value = df_integralEffect

    if "Прогноз" in new_wb.sheets:
        xw.Sheet["Прогноз"].delete()
    new_wb.sheets.add("Прогноз")
    sht = new_wb.sheets("Прогноз")
    sht.range('A1').options().value = df_forecasts

    if "Координаты" in new_wb.sheets:
        xw.Sheet["Координаты"].delete()
    new_wb.sheets.add("Координаты")
    sht = new_wb.sheets("Координаты")
    sht.range('A1').options().value = df_Coordinates

    new_wb.save(str(os.path.basename(data_file)).replace(".xlsx", "") + "_out.xlsx")
    app1.kill()
    # End print
    pass
