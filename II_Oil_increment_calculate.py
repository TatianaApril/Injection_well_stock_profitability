import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from Production_Gain import calculate_production_gain
from Schema import sample_df_one_prod_well
from Utility_function import adding, func


def calculate_oil_increment(df_prod_horizon, last_data, horizon, df_injCells_horizon):
    """
    Расчет НДД с каждой нагнетательной скважины
    :param df_prod_horizon: история добывающих скважин на объекте
    :param last_data: последняя фактиечкая дата
    :param horizon: объект
    :param df_injCells_horizon: таблица с привязкой скважин к ячейкам
    :return: df_final_prod_well, dict_averaged_effects, dict_uncalculated_cells
    """
    list_inj_wells = list(df_injCells_horizon["Ячейка"].unique())
    df_final_prod_well = pd.DataFrame()  # df for all prod well

    # create dictionary for calculating shares for each object
    dict_averaged_effects = dict.fromkeys(df_injCells_horizon["Объект"].unique(), {'Qliq_fact, tons/day': [0],
                                                                                   'Qoil_fact, tons/day': [0],
                                                                                   'delta_Qliq, tons/day': [0],
                                                                                   'delta_Qoil, tons/day': [0]})
    # create dictionary for uncalculated cells
    dict_uncalculated_cells = dict.fromkeys(list_inj_wells, [])

    for inj_well in tqdm(list_inj_wells, desc='II. oil increment'):
        # parameters of inj well
        list_wells_cell = df_injCells_horizon.loc[(df_injCells_horizon["Ячейка"]
                                                   == inj_well)]["№ добывающей"].to_list()
        min_start_date = df_injCells_horizon["Дата запуска ячейки"].min()
        start_date_inj = df_injCells_horizon.loc[(df_injCells_horizon["Ячейка"]
                                                  == inj_well)]["Дата запуска ячейки"].iloc[0]
        for prod_well in list_wells_cell:
            #  sample of Dataframe: df_one_prod_well
            df_one_prod_well = sample_df_one_prod_well.copy()
            slice_well_prod = df_prod_horizon.loc[df_prod_horizon.Well_number == prod_well].reset_index(
                drop=True)

            name_coefficient = "Куч Итог"
            coefficient_prod_well = df_injCells_horizon.loc[(df_injCells_horizon["Ячейка"] == inj_well)
                                                            & (df_injCells_horizon["№ добывающей"] == prod_well
                                                               )][name_coefficient].iloc[0]

            slice_well_prod.loc[:, ("Rate_oil", "Rate_fluid", 'Production_oil', 'Production_fluid')] = \
                slice_well_prod.loc[:, ("Rate_oil", "Rate_fluid", 'Production_oil', 'Production_fluid')] \
                * coefficient_prod_well

            df_one_prod_well.insert(0, "№ добывающей", prod_well)
            df_one_prod_well.insert(0, "Ячейка", inj_well)

            # add columns of date
            df_one_prod_well[pd.date_range(start=min_start_date, end=last_data, freq='MS')] = np.NAN
            df_one_prod_well.loc[0, 3:] = df_one_prod_well.columns[3:]

            # Calculate increment for each prod well________________________________________________________
            slice_well_gain = calculate_production_gain(slice_well_prod, start_date_inj)
            marker_arps = slice_well_gain[2]
            marker = slice_well_gain[1]
            slice_well_gain = slice_well_gain[0]
            slice_well_gain = slice_well_gain[slice_well_gain.Date >= start_date_inj]
            slice_well_gain = slice_well_gain.set_index("Date")

            #  add dictionary for calculating shares for each object
            if marker_arps != "model don't fit":
                object_prod_well = horizon
                dict_df = dict_averaged_effects[object_prod_well].copy()
                for key in dict_df.keys():
                    dict_df[key] = adding(dict_df[key], slice_well_gain[key].values)
                dict_averaged_effects[object_prod_well] = dict_df

                # Проверка типовой кривой
                """ 
                import pickle
                dict_coef = pickle.load(open("file_coef.pkl", 'rb'))
                dict_coef_exp = pickle.load(open("file_coef_exp.pkl", 'rb'))

                import matplotlib.pyplot as plt


                def func_1(x, a, b, c):
                    return a * np.exp(-b * x) + c


                def func(x, a, b):
                    return b * np.exp(-a * np.sqrt(x)) - b

                xdata = range(slice_well_gain.shape[0])
                plt.clf()
                plt.plot(xdata, slice_well_gain['delta_Qliq, tons/day'], c='b')
                plt.plot(xdata, slice_well_gain['delta_Qoil, tons/day'], c='m')

                popt1 = dict_coef[object_prod_well]
                plt.plot(xdata, func(xdata, *popt1[0:2])*slice_well_gain['Qliq_fact, tons/day'], linestyle='--', c='b')
                plt.plot(xdata, func(xdata, *popt1[2:4])*slice_well_gain['Qoil_fact, tons/day'], linestyle='--', c='m')

                popt2 = dict_coef_exp[object_prod_well]
                plt.plot(xdata, func(xdata, *popt2[0:2]) * slice_well_gain['Qliq_fact, tons/day'], linestyle=':',
                         c='b')
                plt.plot(xdata, func(xdata, *popt2[2:4]) * slice_well_gain['Qoil_fact, tons/day'], linestyle=':',
                         c='m')

                plt.savefig(f'pictures/well_{prod_well}_{str(object_prod_well).replace("/", "")}.png', dpi=400,
                            quality=90)
                #plt.show()"""

                for column in slice_well_gain.columns:
                    position = list(slice_well_gain.columns).index(column) + 1
                    df_one_prod_well.iloc[position, 3:] = slice_well_gain[column] \
                        .combine_first(df_one_prod_well.iloc[position, 3:])
                    if column == "accum_liquid_fact":
                        df_one_prod_well.iloc[position, 3:] = df_one_prod_well.iloc[position, 3:].ffill(axis=0)
                df_one_prod_well = df_one_prod_well.fillna(0)
                df_one_prod_well.insert(2, "Статус", marker)
                df_one_prod_well.insert(3, "Арпс/Полка", marker_arps)
                df_one_prod_well.insert(2, "Объект", horizon)
                df_final_prod_well = pd.concat([df_final_prod_well, df_one_prod_well],
                                               axis=0, sort=False).reset_index(drop=True)
            else:
                dict_uncalculated_cells[inj_well] = dict_uncalculated_cells[inj_well] + [prod_well]

    # parts of oil and liquid by object
    averaged_coef = pd.DataFrame(columns=['a_1', 'b_1', 'a_2', 'b_2'])
    for key in dict_averaged_effects.keys():
        dict_averaged_effects[key] = pd.DataFrame(dict_averaged_effects[key])
        dict_averaged_effects[key]["part_liq"] = dict_averaged_effects[key]['delta_Qliq, tons/day'] \
                                                 / dict_averaged_effects[key]['Qliq_fact, tons/day']
        dict_averaged_effects[key]["part_oil"] = dict_averaged_effects[key]['delta_Qoil, tons/day'] \
                                                 / dict_averaged_effects[key]['Qoil_fact, tons/day']

        ydata = dict_averaged_effects[key]["part_liq"]
        xdata = range(dict_averaged_effects[key]["part_liq"].shape[0])
        if ydata.isnull().values.any():
            dict_averaged_effects[key] = [0, 0, 0, 0]
        else:
            popt1, pcov1 = curve_fit(func, xdata, ydata, maxfev=100000)
            ydata = dict_averaged_effects[key]["part_oil"].fillna(0)
            popt2, pcov2 = curve_fit(func, xdata, ydata, maxfev=100000)
            popt = list(popt1) + list(popt2)
            dict_averaged_effects[key] = popt
            averaged_coef = averaged_coef.append(pd.Series(popt,
                                                           index=averaged_coef.columns,
                                                           name=averaged_coef.shape[0]))
    averaged_coef = averaged_coef.mean(axis=0).values
    dict_averaged_effects["Среднее"] = averaged_coef

    return df_final_prod_well, dict_averaged_effects, dict_uncalculated_cells
