from tqdm import tqdm
import pandas as pd
import numpy as np

from Production_Gain import calculate_production_gain
from Schema import sample_df_one_inj_well, sample_df_one_prod_well


def final_adaptation_and_summation(df_prod_horizon, df_inj_horizon, df_final_prod_well,
                                   last_data, horizon, df_injCells_horizon, dict_uncalculated_cells,
                                   dict_averaged_effects, conversion_factor):
    """
    Адаптация нерасчитанных добывающих скважин и суммирование эффекта для нагнетательных скважин
    :param df_final_prod_well: таблица с приростами добывающих
    :param df_inj_horizon: история нагнетательных скважин на объекте
    :param df_prod_horizon: история добывающих скважин на объекте
    :param last_data: последняя фактическая дата
    :param horizon: объект
    :param df_injCells_horizon: таблица с привязкой скважин к ячейкам
    :param dict_uncalculated_cells: словарь с нерасчитанными добывающими в ячейках
    :param dict_averaged_effects: словарь со средними эффектами на объекты
    :param conversion_factor: переводной коэффициент из м3 в т
    :return: df_final_inj_well, df_final_prod_well
    """
    list_inj_wells = list(df_injCells_horizon["Ячейка"].unique())
    df_final_inj_well = pd.DataFrame()  # df for all inj well

    for inj_well in tqdm(list_inj_wells, desc='III. Adaptation of uncalculated wells'):
        # parameters of inj well
        slice_well_inj = df_inj_horizon.loc[df_inj_horizon.Well_number == inj_well].reset_index(drop=True)
        min_start_date = df_injCells_horizon["Дата запуска ячейки"].min()
        start_date_inj = df_injCells_horizon.loc[(df_injCells_horizon["Ячейка"]
                                                  == inj_well)]["Дата запуска ячейки"].iloc[0]

        #  sample of Dataframe: df_one_inj_well
        df_one_inj_well = sample_df_one_inj_well.copy()

        if dict_uncalculated_cells[inj_well]:
            for prod_well in dict_uncalculated_cells[inj_well]:
                #  sample of Dataframe: df_one_prod_well
                df_one_prod_well = sample_df_one_prod_well.copy()
                slice_well_prod = df_prod_horizon.loc[df_prod_horizon.Well_number == prod_well].reset_index(drop=True)

                name_coefficient = "Куч Итог"
                coefficient_prod_well = df_injCells_horizon.loc[(df_injCells_horizon["Ячейка"] == inj_well)
                                                                & (df_injCells_horizon[
                                                                       "№ добывающей"] == prod_well
                                                                   )][name_coefficient].iloc[0]

                slice_well_prod.loc[:, ("Rate_oil", "Rate_fluid", 'Production_oil', 'Production_fluid')] = \
                    slice_well_prod.loc[:, ("Rate_oil", "Rate_fluid", 'Production_oil', 'Production_fluid')] \
                    * coefficient_prod_well

                df_one_prod_well.insert(0, "№ добывающей", prod_well)
                df_one_prod_well.insert(0, "Ячейка", inj_well)

                # add columns of date
                df_one_prod_well[pd.date_range(start=min_start_date, end=last_data, freq='MS')] = np.NAN
                df_one_prod_well.loc[0, 3:] = df_one_prod_well.columns[3:]

                # Calculate increment for each prod well - average______________________________________________
                object_prod_well = horizon
                if sum(dict_averaged_effects[object_prod_well]) == 0:
                    list_aver = dict_averaged_effects["Среднее"]
                else:
                    list_aver = dict_averaged_effects[object_prod_well]
                slice_well_gain = calculate_production_gain(slice_well_prod, start_date_inj, "aver", list_aver)
                marker_arps = slice_well_gain[2]
                marker = slice_well_gain[1]
                slice_well_gain = slice_well_gain[0].set_index("Date")

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

        # add cell sum in df_one_inj_well
        df_one_inj_well.insert(0, "Ячейка", inj_well)

        # add columns of date
        df_one_inj_well[pd.date_range(start=min_start_date, end=last_data, freq='MS')] = np.NAN
        df_one_inj_well.loc[0, 2:] = df_one_inj_well.columns[2:]

        df_one_inj_well.iloc[1, 2:] = df_final_prod_well[(df_final_prod_well["Ячейка"] == inj_well) &
                                                         (df_final_prod_well[
                                                              "Параметр"] == 'Qliq_fact, tons/day')
                                                         ].sum(axis=0).iloc[6:]
        df_one_inj_well.iloc[2, 2:] = df_final_prod_well[(df_final_prod_well["Ячейка"] == inj_well) &
                                                         (df_final_prod_well[
                                                              "Параметр"] == 'Qoil_fact, tons/day')
                                                         ].sum(axis=0).iloc[6:]
        df_one_inj_well.iloc[3, 2:] = df_final_prod_well[(df_final_prod_well["Ячейка"] == inj_well) &
                                                         (df_final_prod_well[
                                                              "Параметр"] == 'delta_Qliq, tons/day')
                                                         ].sum(axis=0).iloc[6:]
        df_one_inj_well.iloc[4, 2:] = df_final_prod_well[(df_final_prod_well["Ячейка"] == inj_well) &
                                                         (df_final_prod_well[
                                                              "Параметр"] == 'delta_Qoil, tons/day')
                                                         ].sum(axis=0).iloc[6:]
        df_one_inj_well.iloc[5, 2:] = df_final_prod_well.loc[(df_final_prod_well["Ячейка"] == inj_well) &
                                                             (df_final_prod_well[
                                                                  "Параметр"] == 'Qliq_fact, tons/day'
                                                              )].dropna(axis=1).astype(bool).sum()[6:]
        series_injection = slice_well_inj[["Injection_rate", "Date"]].set_index("Date")["Injection_rate"]
        df_one_inj_well.iloc[6, 2:] = series_injection.combine_first(df_one_inj_well.iloc[6, 2:])

        df_one_inj_well.iloc[7, 2:] = round((df_one_inj_well.iloc[6, 2:] * conversion_factor)
                                            .div(df_one_inj_well.iloc[1, 2:].where(df_one_inj_well.iloc[1, 2:] != 0,
                                                                                   np.nan)).fillna(0) * 100, 0)

        df_one_inj_well.iloc[8, 2:] = df_final_prod_well.loc[(df_final_prod_well["Ячейка"] == inj_well) &
                                                             (df_final_prod_well[
                                                                  "Параметр"] == "Сumulative fluid production, tons"
                                                              )].sum(axis=0).iloc[6:]



        series_injection_accum = slice_well_inj[["Injection", "Date"]].set_index("Date").cumsum()["Injection"]
        df_one_inj_well.iloc[9, 2:] = series_injection_accum.combine_first(df_one_inj_well.iloc[9, 2:]).ffill(axis=0)

        df_one_inj_well.iloc[10, 2:] = round((df_one_inj_well.iloc[9, 2:] * conversion_factor)
                                             .div(df_one_inj_well.iloc[8, 2:]
                                                  .where(df_one_inj_well.iloc[8, 2:] != 0, np.nan)).fillna(0) * 100, 0)
        df_one_inj_well.insert(1, "тек. Комп на посл. месяц, %", df_one_inj_well.iloc[7, -1])
        df_one_inj_well.insert(1, "накоп. Комп на посл. месяц, %", df_one_inj_well.iloc[10, -1])
        df_one_inj_well.insert(1, "Объект", horizon)
        df_one_inj_well = df_one_inj_well.fillna(0)
        df_final_inj_well = pd.concat([df_final_inj_well, df_one_inj_well], axis=0, sort=False) \
            .reset_index(drop=True)

    return df_final_inj_well, df_final_prod_well
