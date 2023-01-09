import numpy as np
import pandas as pd
from Utility_functionsArps import Calc_FFP


def GainCell_Arps(slice_wellProd, start_date, oilProduction, fluidProduction, timeProduction, nameDate):
    """____________________Апроксимация кривой добычи жидкости_____________________"""
    conf = 0
    dob_zhid = np.array(slice_wellProd[fluidProduction], dtype='float')
    vrem_dob = np.array(slice_wellProd[timeProduction], dtype='float')
    index_start = slice_wellProd[slice_wellProd[nameDate] <= start_date].index.tolist()[-1]
    results = Calc_FFP(dob_zhid[:index_start + 1], vrem_dob[:index_start + 1], conf)

    # Создание профиля
    k1, k2, num_m, Qst = results[:4]
    marker = "Арпс"
    if type(k1) == str:
        k1 = 0
        k2 = 1
        marker = "Полка"
    if k1 < 0:
        k1 = 0
    if k2 < 0:
        k2 = 0

    Qliq = []
    size = slice_wellProd.shape[0] - index_start
    blocks = pd.DataFrame(index=range(size), dtype=object)
    blocks[nameDate] = list(slice_wellProd[nameDate].iloc[index_start:])

    blocks['Qliq_fact, т/сут'] = list(slice_wellProd[fluidProduction].iloc[index_start:]/(
                                   slice_wellProd[timeProduction].iloc[index_start:]/24))

    blocks['Qoil_fact, т/сут'] = list(slice_wellProd[oilProduction].iloc[index_start:] /(
                                   slice_wellProd[timeProduction].iloc[index_start:] / 24))
    blocks['Обводненность, д. ед.'] = 1 - blocks['Qoil_fact, т/сут'] / blocks['Qliq_fact, т/сут']

    for i in range(size):
        Qliq.append(Qst * (1 + k1 * k2 * (num_m - 2)) ** (-1 / k2))
        num_m += 1
    blocks['Qliq_before_inj, т/сут'] = Qliq
    blocks['delta_Qliq, т/сут'] = blocks['Qliq_fact, т/сут'] - blocks['Qliq_before_inj, т/сут']
    blocks['delta_Qliq, т/сут'] = np.where((blocks['delta_Qliq, т/сут'] < 0), 0, blocks['delta_Qliq, т/сут'])
    blocks['delta_Qn, т/сут'] = blocks['delta_Qliq, т/сут'] * (1 - blocks['Обводненность, д. ед.'])
    blocks = blocks[blocks[nameDate] >= start_date]
    blocks["Арпс/Полка"] = marker
    return blocks


def GainOil_Arps(slice_wellProd, start_date, oilProduction, fluidProduction, timeProduction, nameDate):
    """____________________Апроксимация кривой прироста дебита_____________________"""
    conf = 0
    dob_zhid = np.array(slice_wellProd[fluidProduction], dtype='float')
    vrem_dob = np.array(slice_wellProd[timeProduction], dtype='float')
    index_start = slice_wellProd[slice_wellProd[nameDate] == start_date].index.tolist()[-1]
    results = Calc_FFP(dob_zhid[:index_start + 1], vrem_dob[:index_start + 1], conf)

    # Создание профиля
    k1, k2, num_m, Qst = results[:4]
    marker = "Арпс"
    if type(k1) == str:
        k1 = 0
        k2 = 1
        marker = "Полка"
    if k1 < 0:
        k1 = 0
    if k2 < 0:
        k2 = 0

    Qliq = []
    size = slice_wellProd.shape[0] - index_start
    blocks = pd.DataFrame(index=range(size), dtype=object)
    blocks[nameDate] = list(slice_wellProd[nameDate].iloc[index_start:])

    blocks['Qliq_fact, т/сут'] = list(slice_wellProd[fluidProduction].iloc[index_start:]/(
                                   slice_wellProd[timeProduction].iloc[index_start:]/24))

    blocks['Qoil_fact, т/сут'] = list(slice_wellProd[oilProduction].iloc[index_start:] /(
                                   slice_wellProd[timeProduction].iloc[index_start:] / 24))
    blocks['Обводненность, д. ед.'] = 1 - blocks['Qoil_fact, т/сут'] / blocks['Qliq_fact, т/сут']

    for i in range(size):
        Qliq.append(Qst * (1 + k1 * k2 * (num_m - 2)) ** (-1 / k2))
        num_m += 1
    blocks['Qliq_before_inj, т/сут'] = Qliq
    blocks['delta_Qliq, т/сут'] = blocks['Qliq_fact, т/сут'] - blocks['Qliq_before_inj, т/сут']
    blocks['delta_Qn, т/сут'] = blocks['delta_Qliq, т/сут'] * (1 - blocks['Обводненность, д. ед.'])
    blocks["Арпс/Полка"] = marker
    return blocks