import pandas as pd
import numpy as np
import os
from Utility_function_inj import history_processing, find_linear_end


def characteristic_of_desaturation(data_slice, start_date, nameDate, oilProduction, fluidProduction, timeProduction):
    data_slice["accum_liquid"] = data_slice[fluidProduction].cumsum()
    data_slice["accum_oil"] = data_slice[oilProduction].cumsum()
    data_slice["ln_accum_liquid"] = np.log(data_slice["accum_oil"])
    data_slice["Прирост по ХВ, т"] = 0
    data_slice["Прирост по ХВ, т/сут"] = 0
    index_start = data_slice[data_slice[nameDate] <= start_date].index.tolist()[-1]
    slice_base = data_slice.loc[:index_start]
    a, b, model = find_linear_end(slice_base["ln_accum_liquid"], slice_base["accum_oil"])
    if a != 0:
        data_slice["Прирост по ХВ, т"].iloc[index_start:] = \
            list(data_slice["accum_oil"].iloc[index_start:].values.reshape(-1, 1) - \
                 model.predict(data_slice["ln_accum_liquid"].iloc[index_start:].values.reshape(-1, 1)))
        data_slice["Прирост по ХВ, т/сут"].iloc[index_start:] = list(data_slice["Прирост по ХВ, т"].iloc[index_start:] -
                                                                     data_slice["Прирост по ХВ, т"].iloc[index_start])
        data_slice["Прирост по ХВ, т/сут"].iloc[index_start + 2:] = \
            list((data_slice["Прирост по ХВ, т/сут"].iloc[index_start + 2:].reset_index(drop=True) -
                  data_slice["Прирост по ХВ, т/сут"].iloc[index_start +
                                                          1:int(data_slice.shape[0]) - 1].reset_index(drop=True)))
        data_slice["Прирост по ХВ, т/сут"] = data_slice["Прирост по ХВ, т/сут"] / (data_slice[timeProduction] / 24)
        data_slice["Прирост по ХВ, т/сут"] = np.where((data_slice["Прирост по ХВ, т/сут"] < 0), 0,
                                                      data_slice["Прирост по ХВ, т/сут"])
    return data_slice


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
        slice_well_calc = characteristic_of_desaturation(slice_well, Start_date[i]["Дата запуска ППД"],
                                                         nameDate, oilProduction, fluidProduction, timeProduction)
        df_calc = df_calc.append(slice_well_calc)
    pass
