import pandas as pd
import numpy as np
import os
import xlwings as xw
from tqdm import tqdm

from Arps_calculation import Calc_FFP
from Utility_function import history_processing, find_linear_model, func
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
    num_month = data_slice.loc[data_slice.nameDate < start_date].shape[0]
    if data_slice[data_slice.nameDate <= start_date].empty:
        index_start = 0
    else:
        index_start = data_slice[data_slice.nameDate <= start_date].index.tolist()[-1]

    # df_result
    df_result = pd.DataFrame(dtype=object)
    df_result["nameDate"] = data_slice.nameDate.iloc[index_start:].values
    df_result['Qliq_fact, tons/day'] = np.round((data_slice.fluidProduction /
                                                 (data_slice.timeProduction / 24))[index_start:].values, 3)
    df_result['Qoil_fact, tons/day'] = np.round((data_slice.oilProduction /
                                                 (data_slice.timeProduction / 24))[index_start:].values, 3)
    data_slice["accum_liquid"] = data_slice.fluidProduction.cumsum()
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

    if option == "stat":

        # preparation of axes for the calculation
        data_slice["accum_oil"] = data_slice.oilProduction.cumsum()
        data_slice["ln_accum_liquid"] = np.log(data_slice.accum_oil)

        if marker == f'до ППД отработала {str(num_month)} месяцев':

            # liner model characteristic of desaturation
            slice_base = data_slice.loc[:index_start]
            cumulative_oil_base = slice_base.oilProduction[:-1].sum()
            a, b, model = find_linear_model(slice_base.ln_accum_liquid, slice_base.accum_oil)

            # Liquid Production Curve Approximation (Arps)
            production = np.array(data_slice.fluidProduction, dtype='float')[:index_start + 1]
            time_production = np.array(data_slice.timeProduction, dtype='float')[:index_start + 1]
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
                                                  (data_slice.timeProduction[index_start:].values / 24)
                                                  ).cumsum() + cumulative_oil_base

                df_result["accum_oil_base"] = model.predict(np.log(df_result.accum_liquid_base).values.reshape(-1, 1))

                df_result['delta_accum_oil'] = df_result.accum_oil - df_result.accum_oil_base

                df_result['delta_Qoil, tons/day'] = (
                            df_result.delta_accum_oil - df_result.delta_accum_oil.iloc[0]).values
                df_result['delta_Qoil, tons/day'].iloc[1:] = df_result['delta_Qoil, tons/day'][1:].values \
                                                             - df_result['delta_Qoil, tons/day'][:-1].values
                df_result['delta_Qoil, tons/day'] = df_result['delta_Qoil, tons/day'] / \
                                                    (data_slice.timeProduction[index_start:].values / 24)
                df_result['delta_Qoil, tons/day'] = np.where((df_result['delta_Qoil, tons/day'] < 0), 0,
                                                             df_result['delta_Qoil, tons/day'])
                df_result['delta_Qoil, tons/day'] = np.round(df_result['delta_Qoil, tons/day'].values, 3)
                marker = f"{marker}: successful solving"
            else:
                marker = f"{marker}: model don't fit"
    else:
        marker_arps = "по среднему"
        xdata = range(df_result.shape[0])

        df_result['delta_Qliq, tons/day'] = func(xdata, *list_aver[0:2]) * df_result['Qliq_fact, tons/day']
        df_result['delta_Qoil, tons/day'] = func(xdata, *list_aver[2:4]) * df_result['Qoil_fact, tons/day']

    df_result = df_result[["nameDate", 'Qliq_fact, tons/day', 'Qoil_fact, tons/day',
                               'delta_Qliq, tons/day', 'delta_Qoil, tons/day', "accum_liquid_fact"]]
    return [df_result, marker, marker_arps]


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')

    data_file = "files/СевПокур_all.xlsx"
    df_injCells = pd.read_excel(os.path.join(os.path.dirname(__file__), "files/СевПокур_all_out.xlsx"),
                                sheet_name="Ячейки_Северо-Покурское")
    # upload MonthlyOperatingReport
    df_MonthlyOperatingReport = pd.read_excel(os.path.join(os.path.dirname(__file__), data_file),
                                              sheet_name="МЭР").fillna(0)
    # rename columns
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
    df_MonthlyOperatingReport.columns = dict_names_column.values()
    df_MonthlyOperatingReport.wellNumberColumn = df_MonthlyOperatingReport.wellNumberColumn.astype('str')
    df_MonthlyOperatingReport.workHorizon = df_MonthlyOperatingReport.workHorizon.astype('str')
    PROD_MARKER: str = "НЕФ"
    INJ_MARKER: str = "НАГ"
    time_work_min = 5  # minimum well's operation time per month?, days
    df_MonthlyOperatingReport = history_processing(df_MonthlyOperatingReport, PROD_MARKER, INJ_MARKER, time_work_min)
    df_MOR_reservoir = df_MonthlyOperatingReport
    last_data = pd.Timestamp(np.sort(df_MonthlyOperatingReport.nameDate.unique())[-1])

    df_one_inj_well = pd.DataFrame()  # df for one inj well
    df_one_prod_well = pd.DataFrame()  # df for one prod well
    df_final_inj_well = pd.DataFrame()  # df for all inj well
    df_final_prod_well = pd.DataFrame()  # df for all prod well
    listWellsInj = list(df_injCells["Ячейка"].unique())

    # df_result
    df_result_1, df_result_2, df_result_3 = pd.DataFrame(dtype=object), \
                                            pd.DataFrame(dtype=object), \
                                            pd.DataFrame(dtype=object)
    for wellNumberInj in tqdm(listWellsInj, desc='II. oil increment'):

        slice_well_inj = df_MOR_reservoir.loc[df_MOR_reservoir.wellNumberColumn ==
                                              wellNumberInj].reset_index(drop=True)
        slice_well_inj["Injection, m3/day"] = slice_well_inj.waterInjection / \
                                              (slice_well_inj.timeInjection / 24)
        list_wells = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)]["№ добывающей"].to_list()
        min_start_date = df_injCells["Дата запуска ячейки"].min()
        start_date_inj = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)]["Дата запуска ячейки"].iloc[0]

        for prod_well in list_wells:
            slice_well_prod = df_MOR_reservoir.loc[df_MOR_reservoir.wellNumberColumn
                                                   == str(prod_well)].reset_index(drop=True)

            name_coefficient = "Куч доб Итог"
            coefficient_prod_well = df_injCells.loc[(df_injCells["Ячейка"] == wellNumberInj)
                                                    & (df_injCells["№ добывающей"] == prod_well
                                                       )][name_coefficient].iloc[0]

            slice_well_prod.loc[:, ("oilProduction", "fluidProduction")] = \
                slice_well_prod.loc[:, ("oilProduction", "fluidProduction")] * coefficient_prod_well

            # Calculate increment for each prod well________________________________________________________
            data_slice = slice_well_prod
            Qliq_fact = np.round((data_slice.fluidProduction /
                                  (data_slice.timeProduction / 24)).values, 3)
            start_date = start_date_inj
            # number of months before injection
            num_month = data_slice.loc[data_slice.nameDate < start_date].shape[0]
            if data_slice[data_slice.nameDate <= start_date].empty:
                index_start = 0
            else:
                index_start = data_slice[data_slice.nameDate <= start_date].index.tolist()[-1]

            marker_arps = "model don't fit"
            if num_month == 0:
                marker = 'запущена после ППД'
                df3 = pd.DataFrame({str(prod_well): Qliq_fact})
                df_result_3 = pd.concat([df_result_3, df3], ignore_index=False, axis=1)
            elif num_month <= 3:
                marker = 'меньше 4х месяцев работы до ППД'
                df3 = pd.DataFrame({str(prod_well): Qliq_fact})
                df_result_3 = pd.concat([df_result_3, df3], ignore_index=False, axis=1)
            else:
                # injector start index
                index_start = data_slice[data_slice.nameDate <= start_date].index.tolist()[-1]
                if index_start in data_slice.index.tolist()[-3:]:
                    marker = f'После запуска ППД меньше 3х месяцев'
                    df3 = pd.DataFrame({str(prod_well): Qliq_fact})
                    df_result_3 = pd.concat([df_result_3, df3], ignore_index=False, axis=1)
                else:
                    marker = f'до ППД отработала {str(num_month)} месяцев'

                    # preparation of axes for the calculation
                    data_slice["accum_oil"] = data_slice.oilProduction.cumsum()
                    data_slice["ln_accum_liquid"] = np.log(data_slice.accum_oil)

                    if marker == f'до ППД отработала {str(num_month)} месяцев':

                        # liner model characteristic of desaturation
                        slice_base = data_slice.loc[:index_start]

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
                            df3 = pd.DataFrame({str(prod_well): Qliq_fact})
                            df_result_3 = pd.concat([df_result_3, df3], ignore_index=False, axis=1)
                        else:
                            Qliq = []
                            size = data_slice.shape[0] - index_start
                            for month in range(size):
                                Qliq.append(Qst * (1 + k1 * k2 * (num_m - 2)) ** (-1 / k2))
                                num_m += 1

                            df1 = pd.DataFrame({str(prod_well): Qliq})
                            df2 = pd.DataFrame({str(prod_well): list(Qliq_fact[:index_start]) + Qliq})

                            df_result_1 = pd.concat([df_result_1, df1], ignore_index=False, axis=1)
                            df_result_2 = pd.concat([df_result_2, df2], ignore_index=False, axis=1)
                            marker = f"{marker}: successful solving"

    dict_df = {f"1": df_result_1, f"2": df_result_2, f"3": df_result_3}

    # Start print in Excel
    app1 = xw.App(visible=False)
    new_wb = xw.Book()

    for key in dict_df.keys():
        if f"{key}" in new_wb.sheets:
            xw.Sheet[f"{key}"].delete()
        new_wb.sheets.add(f"{key}")
        sht = new_wb.sheets(f"{key}")
        sht.range('A1').options().value = dict_df[key]

    new_wb.save(str(os.path.basename(data_file)).replace(".xlsx", "") + "_ТЕМПЫ.xlsx")
    app1.kill()
    # End print
    pass
