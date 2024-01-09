import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from config import HAS_WORKING_HOURS_FOR_THE_LAST_YEAR


def df_Coordinates_prepare(df, min_length_horWell):
    last_data = pd.Timestamp(np.sort(df.Date.unique())[-1])

    # create DataFrame with coordinates for each well
    new_df = df.loc[df.Date == last_data]
    del new_df['Date']

    new_df["length of well T1-3"] = np.sqrt(np.power(new_df.XT3 - new_df.XT1, 2) + np.power(new_df.YT3 - new_df.YT1, 2))
    new_df["well type"] = 0
    new_df.loc[new_df["length of well T1-3"] < min_length_horWell, "well type"] = "vertical"
    new_df.loc[new_df["length of well T1-3"] >= min_length_horWell, "well type"] = "horizontal"
    new_df.loc[new_df["well type"] == "vertical", 'XT3'] = new_df.XT1
    new_df.loc[new_df["well type"] == "vertical", 'YT3'] = new_df.YT1
    return new_df


def history_prepare(history, type_wells, time_work_min):
    """
    Предварительная обработка истории
    :param history: DataFrame с историей
    :param type_wells: тип скважины 'inj'/'prod'
    :param time_work_min: минимальное время работы скважины в месяц
    :return: обработанная история
    """
    history = history.fillna(0)
    history = history[history.Horizon != "ПК"]

    # Произведем обработку в зависимости от типа скважины
    if type_wells == 'prod':
        history['Time_production'] = history['Time_production_1'] + history['Time_production_2']
        del history['Time_production_1']
        del history['Time_production_2']

        history = history[(history.Rate_oil != 0) & (history.Rate_fluid != 0)
                          & (history.Time_production > time_work_min * 24)]
        history['Production_oil'] = history.Rate_oil * history.Time_production / 24
        history['Production_fluid'] = history.Rate_fluid * history.Time_production / 24
    elif type_wells == 'inj':
        history['Time_injection'] = history['Time_injection_1'] + history['Time_injection_2']
        del history['Time_injection_1']
        del history['Time_injection_2']

        history = history[(history.Injection > 1) & (history.Time_injection > time_work_min * 24)]
        # может брать колонку Qж, м3/сут из тех. режима?
        history["Injection_rate"] = history.Injection / history.Time_injection * 24
    else:
        raise ValueError(f"Wrong type of wells! {type_wells}")

    last_data = np.sort(history.Date.unique())[-1]
    unique_wells = history[history.Date == last_data].Well_number.unique()  # Уникальный список скважин
    history = history[history.Well_number.isin(unique_wells)]

    # Если переменная True - оставляем объекты, находившихся в работе за последний год
    if HAS_WORKING_HOURS_FOR_THE_LAST_YEAR:
        start_data_previous_last_year = last_data.astype('M8[ms]').astype('O') - relativedelta(months=12)
        objects = history[(history.Date >= start_data_previous_last_year)].groupby(['Well_number'])['Horizon'].apply(list)
    else:
        # Оставляем объекты на последнюю дату
        # По умолчанию в расчете участвуют объекты работающие на дату оценки
        objects = history[history.Date == last_data].groupby(['Well_number'])['Horizon'].apply(list)

    history = history[history.apply(lambda x: x['Horizon'] in objects[x.Well_number], axis=1)]

    history = history.sort_values(['Well_number', 'Date'], ascending=True)

    return history


def find_linear_model(x, y):
    """
    :x : Значения по оси X
    :y : Значения по оси Y
    :return : Угловой  коэффициент,
              Свободный коэффициент,
              модель
    """
    r2 = 0
    i = 0
    r2_min = 0.95

    #  Отсекаем по точке сначала графика (чтобы исключить выход на режим):
    # Пока ошибка на МНК по точкам не станет меньше максимальной ошибки
    while r2 < r2_min:
        if i == x.size - 2 or x.size <= 2:
            a, b, model = 0, 0, 0
            break

        model = LinearRegression().fit(x[i:].values.reshape(-1, 1), y[i:].values.reshape(-1, 1))
        r2 = r2_score(model.predict(x[i:].values.reshape(-1, 1)), y[i:].values.reshape(-1, 1))
        b = model.intercept_[0]
        a = model.coef_[0][0]
        i += 1
    return a, b, model


def adding(a, b):
    l = sorted((a, b), key=len)
    c = l[1].copy()
    c[:len(l[0])] += l[0]
    return c


def func(x, a, b):
    return b * np.exp(-a * np.sqrt(x)) - b


def merging_sheets(df_injCells_horizon, df_forecasts, dict_reservoir_df, df_exception_wells, conversion_factor):
    """
    Обединение листов
    :return: dict_reservoir_df
    """
    df_injCells = pd.DataFrame(columns=df_injCells_horizon.columns.tolist())
    df_final_prod_well = pd.DataFrame()
    df_final_inj_well = pd.DataFrame()
    df_forecasts = pd.DataFrame(columns=df_forecasts.columns.tolist())

    for key in list(dict_reservoir_df.keys()):
        if "Ячейки" in key:
            df_injCells = df_injCells.append(dict_reservoir_df.pop(key), ignore_index=True)
        elif "Прогноз" in key:
            df_forecasts = df_forecasts.append(dict_reservoir_df.pop(key), ignore_index=True)
        elif "Прирост доб_" in key:
            if df_final_prod_well.empty:
                df_final_prod_well = dict_reservoir_df.pop(key)
            else:
                df_final_prod_well = pd.concat([df_final_prod_well, dict_reservoir_df.pop(key)], sort=False, axis=0)
            df_final_prod_well.loc[df_final_prod_well['Параметр'] == "Date", 6:] = df_final_prod_well.columns[6:]
        elif "Прирост наг_" in key:
            if df_final_inj_well.empty:
                df_final_inj_well = dict_reservoir_df.pop(key)
            else:
                df_final_inj_well = pd.concat([df_final_inj_well, dict_reservoir_df.pop(key)], sort=False, axis=0)
            df_final_prod_well.loc[df_final_prod_well['Параметр'] == "Date", 4:] = df_final_prod_well.columns[4:]
    dict_reservoir_df["Ячейки"] = df_injCells
    dict_reservoir_df["Прирост доб"] = df_final_prod_well.fillna(0)
    dict_reservoir_df["Прирост наг_по объектам"] = df_final_inj_well.fillna(0)
    dict_reservoir_df["Прогноз_по объектам"] = df_forecasts.copy()

    # Обединение прироста по ячейкам
    del df_final_inj_well["тек. Комп на посл. месяц, %"]
    del df_final_inj_well["накоп. Комп на посл. месяц, %"]
    del df_final_inj_well["Объект"]

    df_final_inj_well['Параметр'] = df_final_inj_well['Параметр'].str.replace("%", "")
    df_final_inj_well.loc[df_final_inj_well['Параметр'] == "Date", 2:] = 0
    df_final_inj_well = df_final_inj_well.groupby(['Ячейка', 'Параметр'])[df_final_inj_well.columns[2:]].sum()

    Injection = np.round(
        np.array(df_final_inj_well.xs('Injection, m3/day', level=1, drop_level=False).astype("float64"))
        * conversion_factor /
        np.array(df_final_inj_well.xs('Qliq_fact, tons/day', level=1, drop_level=False).astype("float64")) * 100, 0)

    Сumulative_water_injection = np.round(
        np.array(df_final_inj_well.xs('Сumulative water injection, tons',
                                      level=1, drop_level=False).astype("float64"))
        * conversion_factor /
        np.array(df_final_inj_well.xs('Сumulative fluid production, tons', level=1,
                                      drop_level=False).astype("float64")) * 100, 0)
    df_final_inj_well = df_final_inj_well.reindex(["Date",
                                                   'Qliq_fact, tons/day',
                                                   'Qoil_fact, tons/day',
                                                   "delta_Qliq, tons/day",
                                                   "delta_Qoil, tons/day",
                                                   "Number of working wells",
                                                   "Injection, m3/day",
                                                   "Current injection ratio, ",
                                                   "Сumulative fluid production, tons",
                                                   "Сumulative water injection, tons",
                                                   "Injection ratio, "], level=1, axis=0)
    df_final_inj_well.reset_index(inplace=True)
    df_final_inj_well['Параметр'] = df_final_inj_well['Параметр'].str.replace("Current injection ratio, ",
                                                                              "Current injection ratio, %")
    df_final_inj_well['Параметр'] = df_final_inj_well['Параметр'].str.replace("Injection ratio, ",
                                                                              "Injection ratio, %")

    df_final_inj_well.loc[df_final_inj_well['Параметр'] == "Current injection ratio, %", 2:] = Injection
    df_final_inj_well.loc[df_final_inj_well['Параметр'] == "Injection ratio, %", 2:] = Сumulative_water_injection
    df_final_inj_well.loc[df_final_inj_well['Параметр'] == "Date", 2:] = df_final_inj_well.columns[2:]
    df_final_inj_well.fillna(0, inplace=True)

    dict_reservoir_df["Прирост_наг_суммарный"] = df_final_inj_well

    # Обединение прироста в прогнозе
    del df_forecasts["Маркер"]
    del df_forecasts["Объект"]
    dict_agg = {df_forecasts.columns[1]: 'max'}
    dict_agg_2 = dict.fromkeys(df_forecasts.columns[3:], 'sum')
    df_forecasts = df_forecasts.groupby(['Ячейка', 'Параметр']).agg({**dict_agg, **dict_agg_2})
    df_forecasts.reset_index(inplace=True)

    dict_reservoir_df["Прогноз_суммарный"] = df_forecasts
    dict_reservoir_df["Исключены_из_расчета"] = df_exception_wells

    return dict_reservoir_df


def get_period_of_working_for_calculating(df: pd.DataFrame, months: int) -> pd.DataFrame:
    """
    Подготовка периода работы скважин за последние месяцы от текущей даты
    :param df: - фрейм с данными о работе скважин (нагнетельных или добывающих)
    :param months: - количество последних месяцев, которые необходимо включить в расчет
    :return: - фрейм с данными о работе скважин, которые работали последние Х месяцев от последней даты в МЭР
    """
    last_working_period = df.Date.max() - relativedelta(months=months)
    count_months = df[df.Date >= last_working_period].groupby('Well_number', as_index=False).agg({'Date': 'count'})
    result = df[df.Well_number.isin(list(count_months.Well_number.unique()))]

    return result
