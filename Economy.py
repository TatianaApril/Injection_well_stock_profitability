import os
import sqlite3
import pandas as pd
import numpy as np
import yaml
from dateutil.relativedelta import relativedelta
from loguru import logger
import xlwings as xw

from Economy_functions import select_analogue, expenditure_side, revenue_side, estimated_revenue, \
    taxes, Profit, FCF, DCF

import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == '__main__':

    logger.info(f"upload conf_files")
    with open('conf_files/liquid_groups.yml') as f:
        liquid_groups = pd.DataFrame(yaml.safe_load(f))
    liquid_groups = liquid_groups.astype("float")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    logger.info(f"path:{dir_path}")
    database_path = dir_path + "\\database"
    logger.info(f"Upload database for Экономика")
    connection = sqlite3.connect(database_path + f'//Экономика.db')

    coefficients = pd.read_sql("SELECT * from coefficients", connection)
    macroeconomics = pd.read_sql("SELECT * from macroeconomics", connection)
    df_FPA = pd.read_sql("SELECT * from df_FPA", connection)
    reservoirs_NDD = pd.read_sql("SELECT * from reservoirs_NDD", connection)

    connection.commit()
    connection.close()

    # insert the group of liquid in df_FPA
    df_FPA.iloc[:, 6:] = df_FPA.iloc[:, 6:].astype("float")
    df_FPA.rename(columns={'Дебит жидк., т/сут': "Qliq_min"}, inplace=True)
    df_FPA.sort_values(by=["Qliq_min"], inplace=True)
    df_FPA["liquid_group"] = pd.merge_asof(df_FPA["Qliq_min"],
                                           liquid_groups, on="Qliq_min", direction="nearest").iloc[:, -1]

    logger.info("check the content of output")
    output_path = dir_path + "\\output"
    output_content = os.listdir(path=output_path)
    try:
        output_content.remove('Экономика')
    except ValueError:
        pass

    if output_content:
        logger.info(f"reservoirs: {len(output_content)}")
    else:
        raise FileExistsError("no files!")

    for file in output_content:
        logger.info(f"load file: {file}")
        file_path = output_path + f"\\{file}"
        reservoir = file.replace(".xlsx", "")

        # Dataframes:
        df_prod_well = pd.read_excel(file_path, sheet_name="Прирост доб")
        df_prod_well[['№ добывающей', 'Ячейка']] = df_prod_well[['№ добывающей', 'Ячейка']].astype("str")

        years = pd.Series(df_prod_well.columns[7:]).dt.to_period("A")
        last_year = years.iloc[-1]
        last_data = df_prod_well.columns[-1]
        index_start_year = years[years == last_year].index[0] + 7
        df_prod_well = df_prod_well.iloc[:, [1, 2, 3, 6]].merge(df_prod_well.iloc[:, index_start_year:],
                                                                right_index=True, left_index=True)
        df_prod_well = df_prod_well[df_prod_well["Параметр"].isin(["delta_Qliq, tons/day",
                                                                   "delta_Qoil, tons/day",
                                                                   "Qliq_fact, tons/day"])]
        df_prod_well.insert(loc=0, column="Месторождение", value=reservoir)
        df_Qliq = df_prod_well[df_prod_well["Параметр"] == "delta_Qliq, tons/day"].reset_index(drop=True)
        df_Qoil = df_prod_well[df_prod_well["Параметр"] == "delta_Qoil, tons/day"].reset_index(drop=True)
        df_Qliq_fact = df_prod_well[df_prod_well["Параметр"] == "Qliq_fact, tons/day"].iloc[:, [2, -1]].reset_index(
            drop=True)
        df_Qliq_fact = df_Qliq_fact.groupby("№ добывающей").sum()

        df_forecasts = pd.read_excel(file_path, sheet_name="Прогноз_суммарный").iloc[:, 1:]
        df_forecasts['Ячейка'] = df_forecasts['Ячейка'].astype("str")

        logger.info(f"Объединение факта с прогнозом для добычи")
        if df_forecasts.shape[1] > 3:

            df_forecasts.columns = df_forecasts.columns[:3].to_list() + \
                                   [pd.to_datetime(last_data) + relativedelta(months=i + 1) for i in range(df_forecasts.shape[1] - 3)]
            del df_forecasts["Последняя дата работы"]
            df_forecasts.set_index("Ячейка", inplace=True)

            df_fQliq = df_forecasts[df_forecasts["Параметр"] == "delta_Qliq, tons/day"]
            df_fQoil = df_forecasts[df_forecasts["Параметр"] == "delta_Qoil, tons/day"]

            sum_Qliq = df_Qliq[['Ячейка', last_data]].groupby(by=['Ячейка']).sum()
            sum_Qoil = df_Qoil[['Ячейка', last_data]].groupby(by=['Ячейка']).sum()

            df_ratio_liq = df_Qliq[['Ячейка', "№ добывающей", last_data]]
            df_ratio_oil = df_Qoil[['Ячейка', "№ добывающей", last_data]]

            df_ratio_liq["ratio"] = df_ratio_liq.apply(lambda row: row[last_data] / sum_Qliq.loc[row['Ячейка']],
                                                       axis=1).fillna(0)
            df_ratio_oil["ratio"] = df_ratio_oil.apply(lambda row: row[last_data] / sum_Qoil.loc[row['Ячейка']],
                                                       axis=1).fillna(0)
            for column in df_forecasts.columns[1:]:
                df_Qliq[column] = df_ratio_liq.apply(lambda row: row["ratio"] *
                                                                 df_fQliq[column].loc[row['Ячейка']], axis=1).fillna(0)
                df_Qoil[column] = df_ratio_oil.apply(lambda row: row["ratio"] *
                                                                 df_fQoil[column].loc[row['Ячейка']], axis=1).fillna(0)

        logger.info(f"Проверка наличия всех скважин в НРФ")
        df_FPA.rename(columns={'№скв.': '№ добывающей'}, inplace=True)
        merged_FPA = df_Qliq[["Месторождение", "№ добывающей"]].merge(df_FPA,
                                                                      left_on=["Месторождение", "№ добывающей"],
                                                                      right_on=["Месторождение", "№ добывающей"],
                                                                      how='left')

        columns_start = merged_FPA.columns
        merged_FPA = merged_FPA.apply(
            lambda row: select_analogue(df_FPA.copy(), row, df_Qliq_fact.loc[row["№ добывающей"]][0], liquid_groups)
            if np.isnan(row['liquid_group']) else row, axis=1)
        merged_FPA = merged_FPA[columns_start]

        df_inj_well = pd.read_excel(file_path, sheet_name="Прирост_наг_суммарный")
        df_inj_well[['Ячейка']] = df_inj_well[['Ячейка']].astype("str")

        index_start_year = years[years == last_year].index[0] + 3
        df_inj_well = df_inj_well.iloc[:, [1, 2]].merge(df_inj_well.iloc[:, index_start_year:],
                                                        right_index=True, left_index=True)
        df_W = df_inj_well[df_inj_well["Параметр"] == "Injection, m3/day"].reset_index(drop=True)

        logger.info(f"Объединение факта с прогнозом для закачки")
        if df_forecasts.shape[1] > 3:
            for column in df_forecasts.columns[1:]:
                df_W[column] = df_W.iloc[:, -1]

        unit_costs_oil = merged_FPA["Уделка на нефть, руб/тн.н"]
        unit_costs_injection = df_FPA[df_FPA["Месторождение"] == reservoir]['Уделка на закачку, руб/м3'].mean()
        unit_cost_fluid = merged_FPA['Уделка на жидкость, руб/т']
        unit_cost_water = merged_FPA["Уделка на воду, руб/м3"]
        K_d = merged_FPA["Кд"]

        logger.info(f"Расходная часть")
        all_cost = expenditure_side(df_Qoil, df_Qliq, df_W, unit_costs_oil,
                                    unit_costs_injection, unit_cost_fluid, unit_cost_water)

        Netback = macroeconomics[macroeconomics["Параметр"] == "Netback"]
        Urals = macroeconomics[macroeconomics["Параметр"] == "Urals"]
        dollar_rate = macroeconomics[macroeconomics["Параметр"] == "exchange_rate"]

        logger.info(f"Доходная часть")
        income = revenue_side(Netback, df_Qoil)
        logger.info(f"Расчетная выручка для расчета налога по схеме НДД")
        income_taxes = estimated_revenue(df_Qoil, Urals, dollar_rate)

        export_duty = macroeconomics[macroeconomics["Параметр"] == "customs_duty"]
        cost_transportation = macroeconomics[macroeconomics["Параметр"] == "cost_transportation"]
        K_man = macroeconomics[macroeconomics["Параметр"] == "K_man"]
        K_dt = macroeconomics[macroeconomics["Параметр"] == "K_dt"]

        coefficients_res = coefficients[coefficients["Месторождение"] == reservoir]
        if coefficients_res.empty:
            raise ValueError(f"Wrong name for coefficients: {reservoir}")
        coefficients_res = coefficients_res.values.tolist()[0][1:]

        method = "mineral_extraction_tax"
        if reservoir in reservoirs_NDD.values:
            method = "income_tax_additional"
        K = [coefficients_res, K_man, K_dt, K_d]

        logger.info(f"Налоги")
        all_taxes = taxes(df_Qoil, Urals, dollar_rate, export_duty, cost_transportation, income_taxes,
                                          all_cost,
                                          *K, method=method)
        logger.info(f"Прибыль")
        profit = Profit(income, all_cost, all_taxes)
        logger.info(f"FCF")
        fcf = FCF(profit, profits_tax=0.2)
        r = macroeconomics[macroeconomics["Параметр"] == "r"]
        dcf = DCF(fcf, r)
        npv = dcf.cumsum(axis=1)

        # Выгрузка для Маши
        num_days = np.reshape(np.array(pd.to_datetime(df_Qoil.columns[5:]).days_in_month), (1, -1))
        years = pd.Series(df_Qoil.columns[5:], name='year').dt.year
        incremental_oil_production = np.array(df_Qoil.iloc[:, 5:]) * num_days
        incremental_oil_production = pd.DataFrame(incremental_oil_production, columns=df_Qoil.columns[5:])
        incremental_oil_production = pd.concat([df_Qoil.iloc[:, :4], incremental_oil_production], axis=1)
        incremental_oil_production = incremental_oil_production.groupby(['Ячейка'])[
            incremental_oil_production.columns[4:]].sum().sort_index()

        # Start print in Excel for one reservoir
        app1 = xw.App(visible=False)
        new_wb = xw.Book()


        def add_on_sheet(name, df):
            if name in new_wb.sheets:
                xw.Sheet[name].delete()
            new_wb.sheets.add(name)
            sht = new_wb.sheets(name)
            sht.range('A1').options().value = df
            pass


        """add_on_sheet(f"закачка", df_W)
        add_on_sheet(f"Qн добывающих", df_Qoil)
        add_on_sheet(f"Qж добывающих", df_Qliq)
        add_on_sheet(f"Qн нагнетательных", incremental_oil_production)

        add_on_sheet(f"Прибыль", profit)
        add_on_sheet(f"доходная часть", income)
        add_on_sheet(f"macroeconomics", macroeconomics)
        add_on_sheet(f"НРФ", merged_FPA)
        add_on_sheet(f"коэффициенты из НРФ", coefficients)
        add_on_sheet(f"налоги", all_taxes)
        add_on_sheet(f"затраты", all_cost)
        add_on_sheet(f"затраты на добывающие", cost_prod_wells)
        add_on_sheet(f"затраты на нагнетательные", cost_inj_wells)"""

        add_on_sheet(f"{reservoir} FCF", fcf)
        add_on_sheet(f"{reservoir} DCF", dcf)
        add_on_sheet(f"{reservoir} NPV", npv)

        logger.info(f"Запись .xlsx")
        new_wb.save(dir_path + f"\\output\\Экономика\\{reservoir}_экономика.xlsx")
        app1.kill()

logger.info("good end :)")
