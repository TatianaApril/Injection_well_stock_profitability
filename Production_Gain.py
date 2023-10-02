import pandas as pd
import numpy as np
import os
import xlwings as xw
from tqdm import tqdm

from Arps_calculation import Calc_FFP
from Utility_function import find_linear_model, func
import matplotlib.pyplot as plt


def calculate_production_gain(data_slice, start_date, option="stat", list_aver=[0, 0, 0, 0]):
    """
    Расчет прироста добычи от нагнетательной скважины
    :param list_aver: список коэффциентов для типовой кривой с объекта
    :param option: stat/aver расчет на онове статитики или средних долей по объекту
    :param data_slice: исходная таблица МЭР для добывающей скважины
    :param start_date: начало работы нагнетательной скважины в ячейке
    :return: [df_result, marker]
    """
    # number of months before injection
    num_month = data_slice.loc[data_slice.Date < start_date].shape[0]
    if data_slice[data_slice.Date <= start_date].empty:
        index_start = 0
    else:
        index_start = data_slice[data_slice.Date <= start_date].index.tolist()[-1]

    # df_result
    df_result = pd.DataFrame(dtype=object)
    df_result["Date"] = data_slice.Date.iloc[index_start:]
    df_result['Qliq_fact, tons/day'] = np.round(data_slice.Rate_fluid[index_start:].values, 3)
    df_result['Qoil_fact, tons/day'] = np.round(data_slice.Rate_oil[index_start:].values, 3)
    data_slice["accum_liquid"] = data_slice.Production_fluid.cumsum()
    df_result["accum_liquid_fact"] = data_slice.accum_liquid[index_start:].values

    df_result[['delta_Qliq, tons/day', 'delta_Qoil, tons/day']] = 0

    marker_arps = "model don't fit"
    if num_month == 0:
        marker = 'запущена после ППД'
    elif num_month <= 3:
        marker = 'меньше 4х месяцев работы до ППД'
    else:
        # injector start index
        index_start = data_slice[data_slice.Date <= start_date].index.tolist()[-1]
        if index_start in data_slice.index.tolist()[-3:]:
            marker = f'После запуска ППД меньше 3х месяцев'
        else:
            marker = f'до ППД отработала {str(num_month)} месяцев'

    if option == "stat":

        # preparation of axes for the calculation
        data_slice["accum_oil"] = data_slice.Production_oil.cumsum()
        data_slice["ln_accum_liquid"] = np.log(data_slice.accum_oil)

        if marker == f'до ППД отработала {str(num_month)} месяцев':

            # liner model characteristic of desaturation
            slice_base = data_slice.loc[:index_start]
            cumulative_oil_base = slice_base.Production_oil[:-1].sum()
            a, b, model = find_linear_model(slice_base.ln_accum_liquid, slice_base.accum_oil)

            # Liquid Production Curve Approximation (Arps)
            production = np.array(data_slice.Production_fluid, dtype='float')[:index_start + 1]
            time_production = np.array(data_slice.Time_production, dtype='float')[:index_start + 1]
            """
            plt.clf()
            array_rates = np.array(data_slice.fluidProduction, dtype='float') / (np.array(data_slice.timeProduction, dtype='float') / 24)
            plt.scatter(np.array(data_slice.nameDate), array_rates)
            plt.scatter(np.array(data_slice.nameDate)[index_start], array_rates[index_start], c='red')
            plt.plot(np.array(data_slice.nameDate), array_rates)"""

            results_approximation = Calc_FFP(production, time_production)
            k1, k2, num_m, Qst = results_approximation[:4]
            marker_arps = "Арпс"
            if k1 == 0 and k2 == 1:
                marker_arps = "Полка"
            elif type(k1) == str:
                marker_arps = "model don't fit"

            if a != 0 and k1 != "Невозможно":
                df_result["accum_oil"] = data_slice.accum_oil[index_start:].values
                # recovery of base fluid production
                Qliq = []
                size = data_slice.shape[0] - index_start
                for month in range(size):
                    try:
                        current = Qst * (1 + k1 * k2 * (num_m - 2)) ** (-1 / k2)
                    except ZeroDivisionError:
                        current = 0
                    Qliq.append(current)
                    num_m += 1
                """
                Qliq2 = []
                index = list(np.where(array_rates[:index_start + 1] == np.amax(array_rates[:index_start + 1])))[0][0]
                m=index
                size = data_slice.shape[0] - index
                for m in range(size):
                    Qliq2.append(Qst * (1 + k1 * k2 * (m)) ** (-1 / k2))

                plt.plot(np.array(data_slice.nameDate)[index:], Qliq2, c='red')
                plt.title(f"k1={k1}, k2={k2}")
                plt.savefig(f'pictures/picture_of_{slice_base.wellNumberColumn.values[0]}.png', dpi=70, quality=50)
                #plt.show()"""

                df_result['Qliq_base, tons/day'] = Qliq
                df_result['delta_Qliq, tons/day'] = df_result['Qliq_fact, tons/day'] - df_result['Qliq_base, tons/day']
                df_result['delta_Qliq, tons/day'] = np.where((df_result['delta_Qliq, tons/day'] < 0), 0,
                                                             df_result['delta_Qliq, tons/day'])
                df_result['delta_Qliq, tons/day'] = np.round(df_result['delta_Qliq, tons/day'].values, 3)
                df_result["Арпс/Полка"] = marker_arps

                df_result["accum_liquid_base"] = (df_result['Qliq_base, tons/day'] *
                                                  (data_slice.Time_production[index_start:].values / 24)
                                                  ).cumsum() + cumulative_oil_base

                df_result["accum_oil_base"] = model.predict(np.log(df_result.accum_liquid_base).values.reshape(-1, 1))

                df_result['delta_accum_oil'] = df_result.accum_oil - df_result.accum_oil_base

                df_result['delta_Qoil, tons/day'] = (
                            df_result.delta_accum_oil - df_result.delta_accum_oil.iloc[0]).values
                df_result['delta_Qoil, tons/day'].iloc[1:] = df_result['delta_Qoil, tons/day'][1:].values \
                                                             - df_result['delta_Qoil, tons/day'][:-1].values
                df_result['delta_Qoil, tons/day'] = df_result['delta_Qoil, tons/day'] / \
                                                    (data_slice.Time_production[index_start:].values / 24)
                df_result['delta_Qoil, tons/day'] = np.where((df_result['delta_Qoil, tons/day'] < 0), 0,
                                                             df_result['delta_Qoil, tons/day'])
                df_result['delta_Qoil, tons/day'] = np.round(df_result['delta_Qoil, tons/day'].values, 3)
                marker = f"{marker}: successful solving"
            else:
                marker = f"{marker}: model don't fit"
    elif option == "aver":
        marker_arps = "по среднему"
        xdata = range(df_result.shape[0])

        df_result['delta_Qliq, tons/day'] = func(xdata, *list_aver[0:2]) * df_result['Qliq_fact, tons/day']
        df_result['delta_Qoil, tons/day'] = func(xdata, *list_aver[2:4]) * df_result['Qoil_fact, tons/day']
    else:
        raise AttributeError(f"wrong option: {option}")

    df_result = df_result[["Date", 'Qliq_fact, tons/day', 'Qoil_fact, tons/day',
                               'delta_Qliq, tons/day', 'delta_Qoil, tons/day', "accum_liquid_fact"]]
    return [df_result, marker, marker_arps]

