from tqdm import tqdm
import pandas as pd
import numpy as np

from Arps_calculation import Calc_FFP


def calculate_forecast(list_inj_wells, df_final_inj_well,
                       df_injCells_horizon, horizon, time_predict):
    """
    Расчет прогноза приростов для нагнетательных скважин и объединение эффектов
    :param list_inj_wells: список нагнетательных скважин
    :param df_final_inj_well: таблица с приростами нагнетательных
    :param horizon: объект
    :param df_injCells_horizon: таблица с привязкой скважин к ячейкам
    :param time_predict: период прогноза
    :return: df_forecasts
    """
    df_forecasts = pd.DataFrame()
    for inj_well in tqdm(list_inj_wells, desc='IV. Integral effect and forecast'):
        df_part_forecasts = pd.DataFrame(np.array(["delta_Qliq, tons/day",
                                                   "delta_Qoil, tons/day"]), columns=['Параметр'])

        slice_well_inj = df_final_inj_well.loc[df_final_inj_well["Ячейка"] == inj_well]
        dates_inj = slice_well_inj[slice_well_inj["Параметр"] == 'Date'].iloc[0, 5:]
        start_date_inj = df_injCells_horizon.loc[(df_injCells_horizon["Ячейка"]
                                                  == inj_well)]["Дата запуска ячейки"].iloc[0]

        df_part_forecasts.insert(0, "Маркер", "ok")
        df_part_forecasts.insert(0, "Последняя дата работы", dates_inj.iloc[-1])
        df_part_forecasts.insert(0, "Объект", horizon)
        df_part_forecasts.insert(0, "Ячейка", inj_well)

        df_part_forecasts[list(range(time_predict))] = 0

        df_part_forecasts.iloc[0, 5:] = 0
        df_part_forecasts.iloc[1, 5:] = 0

        delta_Qliq = slice_well_inj[slice_well_inj["Параметр"] ==
                                    'delta_Qliq, tons/day'].iloc[0, 5:][start_date_inj:].values
        delta_Qoil = slice_well_inj[slice_well_inj["Параметр"] ==
                                    'delta_Qoil, tons/day'].iloc[0, 5:][start_date_inj:].values

        parameters = [delta_Qliq, delta_Qoil]
        i = 0
        for parameter in parameters:
            parameter = parameter[parameter != 0]
            if len(parameter) != 0:
                # Arps
                production = np.array(parameter)
                time_production = np.ones(parameter.shape[0]) * 24
                results_approximation = Calc_FFP(production, time_production)
                k1, k2, num_m, Qst = results_approximation[:4]
                if type(k1) == str or k2 == 0:
                    df_part_forecasts.iloc[i, 3] = "error"
                else:
                    if k1 == 0 and k2 == 1:
                        df_part_forecasts.iloc[i, 3] = "полка"
                    rate = []
                    for month in range(time_predict):
                        rate.append(Qst * (1 + k1 * k2 * (num_m - 2)) ** (-1 / k2))
                        num_m += 1
                    df_part_forecasts.iloc[i, 5:] = rate
                i += 1

        df_forecasts = pd.concat([df_forecasts, df_part_forecasts], axis=0, sort=False).reset_index(
            drop=True)
    return df_forecasts
