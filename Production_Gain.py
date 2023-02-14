import pandas as pd
import numpy as np
import os
from Arps_calculation import Calc_FFP
from Utility_function import history_processing, find_linear_model


def calculate_production_gain(data_slice, start_date):
    """
    Расчет прироста добычи от нагнетательной скважины
    :param data_slice: исходная таблица МЭР для добывающей скважины
    :param start_date: начало работы нагнетательной скважины в ячейке
    :return: [df_result, marker]
    """
    # number of months before injection
    num_month = data_slice.loc[data_slice.nameDate < start_date].shape[0]
    if data_slice[data_slice.nameDate <= start_date].empty:
        index_start = 0
    else:
        index_start = data_slice[data_slice.nameDate <= start_date].index.tolist()[-1]

    # preparation of axes for the calculation
    data_slice["accum_liquid"] = data_slice.fluidProduction.cumsum()
    data_slice["accum_oil"] = data_slice.oilProduction.cumsum()
    data_slice["ln_accum_liquid"] = np.log(data_slice.accum_oil)

    # df_result
    df_result = pd.DataFrame(dtype=object)
    df_result["nameDate"] = data_slice.nameDate.iloc[index_start:].values
    df_result['Qliq_fact, tons/day'] = (data_slice.fluidProduction /
                                        (data_slice.timeProduction / 24))[index_start:].values
    df_result['Qoil_fact, tons/day'] = (data_slice.oilProduction /
                                        (data_slice.timeProduction / 24))[index_start:].values
    df_result["accum_liquid_fact"] = data_slice.accum_liquid[index_start:].values

    df_result[['delta_Qliq, tons/day', 'delta_Qoil, tons/day']] = 0

    marker_arps = "model don't fit"
    if num_month == 0:
        marker = 'запущена после ППД'
    elif num_month <= 3:
        marker = 'меньше 4х месяцев работы до ППД'
    else:
        # injector start index
        index_start = data_slice[data_slice.nameDate <= start_date].index.tolist()[-1]
        if index_start in data_slice.index.tolist()[-3:]:
            marker = f'После запуска ППД меньше 3х месяцев'
        else:
            marker = f'до ППД отработала {str(num_month)} месяцев'

            # liner model characteristic of desaturation
            slice_base = data_slice.loc[:index_start]
            cumulative_oil_base = slice_base.oilProduction[:-1].sum()
            a, b, model = find_linear_model(slice_base.ln_accum_liquid, slice_base.accum_oil)

            # Liquid Production Curve Approximation (Arps)
            production = np.array(data_slice.fluidProduction, dtype='float')[:index_start + 1]
            time_production = np.array(data_slice.timeProduction, dtype='float')[:index_start + 1]
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
                    Qliq.append(Qst * (1 + k1 * k2 * (num_m - 2)) ** (-1 / k2))
                    num_m += 1

                df_result['Qliq_base, tons/day'] = Qliq
                df_result['delta_Qliq, tons/day'] = df_result['Qliq_fact, tons/day'] - df_result['Qliq_base, tons/day']
                df_result['delta_Qliq, tons/day'] = np.where((df_result['delta_Qliq, tons/day'] < 0), 0,
                                                             df_result['delta_Qliq, tons/day'])
                df_result["Арпс/Полка"] = marker_arps

                df_result["accum_liquid_base"] = (df_result['Qliq_base, tons/day'] *
                                                  (data_slice.timeProduction[index_start:].values / 24)
                                                  ).cumsum() + cumulative_oil_base

                df_result["accum_oil_base"] = model.predict(np.log(df_result.accum_liquid_base).values.reshape(-1, 1))

                df_result['delta_accum_oil'] = df_result.accum_oil - df_result.accum_oil_base

                df_result['delta_Qoil, tons/day'] = (df_result.delta_accum_oil - df_result.delta_accum_oil.iloc[0]).values
                df_result['delta_Qoil, tons/day'].iloc[1:] = df_result['delta_Qoil, tons/day'][1:].values \
                                                          - df_result['delta_Qoil, tons/day'][:-1].values
                df_result['delta_Qoil, tons/day'] = df_result['delta_Qoil, tons/day'] / \
                                                 (data_slice.timeProduction[index_start:].values / 24)
                df_result['delta_Qoil, tons/day'] = np.where((df_result['delta_Qoil, tons/day'] < 0), 0,
                                                          df_result['delta_Qoil, tons/day'])
                marker = f"{marker}: successful solving"
            else:
                marker = f"{marker}: model don't fit"

    df_result = df_result[["nameDate", 'Qliq_fact, tons/day', 'Qoil_fact, tons/day',
                           'delta_Qliq, tons/day', 'delta_Qoil, tons/day', "accum_liquid_fact"]]
    return [df_result, marker, marker_arps]


if __name__ == '__main__':
    data_file = "Аспид ппд2.xlsx"

    nameDate = 'Дата'
    oilProduction = 'Добыча нефти за посл.месяц, т'
    fluidProduction = 'Добыча жидкости за посл.месяц, т'
    timeProduction = 'Время работы в добыче, часы'

    df_initial = pd.read_excel(os.path.join(os.path.dirname(__file__), data_file), sheet_name="МЭР")
    Start_date = pd.read_excel(os.path.join(os.path.dirname(__file__), data_file), sheet_name="Даты").set_index(
        'Скважина')
    Start_date = Start_date.to_dict('index')
    df_initial = history_processing(df_initial)
    wells_uniq = df_initial['№ скважины'].unique()
    df_calc = pd.DataFrame()
    for i in wells_uniq:
        slice_well = df_initial.loc[df_initial['№ скважины'] == i].copy().reset_index(drop=True)
        slice_well_calc = calculate_production_gain(slice_well, Start_date[i]["Дата запуска ППД"],
                                                    nameDate, oilProduction, fluidProduction, timeProduction)
        df_calc = df_calc.append(slice_well_calc)
    pass
