import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def calculation_coefficients(df_injCelles, df_coeff):
    """
    Расчет коэффициентов участия и влияния
    :param df_injCelles: Исходный массив
    :param df_coeff: массив с табличными понижающими коэффициентами
    :return: отредактированный df_injCelles
    """
    # calculation coefficients
    df_injCelles["Кнаг"] = df_injCelles["Нн, м"] / df_injCelles["Расстояние, м"]
    df_injCelles["Кдоб"] = df_injCelles["Нд, м"] / df_injCelles["Расстояние, м"]
    sum_Kinj = df_injCelles[["Ячейка", "Кнаг"]].groupby(by=["Ячейка"]).sum()
    sum_Kprod = df_injCelles[["№ добывающей", "Кдоб"]].groupby(by=["№ добывающей"]).sum()

    df_injCelles["Куч"] = df_injCelles.apply(lambda row: df_injCelles["Кдоб"].iloc[row.name] /
                                                         sum_Kprod.loc[df_injCelles["№ добывающей"].iloc[row.name]],
                                             axis=1)
    df_injCelles["Квл"] = df_injCelles.apply(lambda row: df_injCelles["Кнаг"].iloc[row.name] /
                                                         sum_Kinj.loc[df_injCelles["Ячейка"].iloc[row.name]],
                                             axis=1)
    df_injCelles["Куч*Квл"] = df_injCelles["Куч"] * df_injCelles["Квл"]

    sum_Kmultiplication = df_injCelles[["№ добывающей", "Куч*Квл"]].groupby(by=["№ добывающей"]).sum()
    df_injCelles["Куч доб"] = df_injCelles.apply(lambda row: df_injCelles["Куч*Квл"].iloc[row.name] /
                                                             sum_Kmultiplication.loc[
                                                                 df_injCelles["№ добывающей"].iloc[row.name]],
                                                 axis=1)
    df_coeff.columns = ["Расстояние, м", "Куч доб табл"]
    df_coeff = df_coeff.astype('float64')

    df_injCelles = df_injCelles.sort_values(by="Расстояние, м").reset_index(drop=True)
    df_merge = pd.merge_asof(df_injCelles["Расстояние, м"],
                             df_coeff.sort_values(by="Расстояние, м"), on="Расстояние, м", direction="nearest")
    df_injCelles["Куч доб Итог"] = df_injCelles["Куч доб"].where(df_injCelles["Куч доб"] != 1, df_merge["Куч доб табл"])
    return df_injCelles


def history_processing(history):
    """
    Предварительная обработка МЭР
    :param history: DataFrame {№ скважины;Дата;Добыча нефти за посл.месяц, т;Добыча жидкости за посл.месяц, т;
    Время работы в добыче, часы;Объекты работы;Координата забоя Y (по траектории);Координата забоя Х (по траектории)}
    :param max_delta Максимальный период остановки, дни
    :return: обрезанный DataFrame
    """
    last_data = history['Дата'].unique()
    last_data.sort(axis=0)
    history = history.fillna(0)  # Заполнение пустых ячеек нулями
    history = history[(history['Добыча нефти за посл.месяц, т'] != 0) &
                      (history['Добыча жидкости за посл.месяц, т'] != 0) &
                      (history['Время работы в добыче, часы'] != 0) &
                      (history['Объекты работы'] != 0)]  # Оставляем не нулевые строки

    unique_wells = history['№ скважины'].unique()  # Уникальный список скважин (без нулевых значений)
    history = history[history['№ скважины'].isin(unique_wells)]  # Выделяем историю только этих скважин
    history = history.sort_values(['№ скважины', 'Дата'])
    history_new = pd.DataFrame()
    for i in unique_wells:
        slice = history.loc[history['№ скважины'] == i].copy()
        object = slice['Объекты работы'].iloc[-1]
        slice = slice[slice['Объекты работы'] == object]
        slice = slice.reset_index()
        history_new = history_new.append(slice, ignore_index=True)
    return history_new


def find_linear_end(x, y):
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
        if i == x.size-2 or x.size <= 2:
            a, b, model = 0, 0, 0
            break
        model = LinearRegression().fit(x[i:].values.reshape(-1, 1), y[i:].values.reshape(-1, 1))
        r2 = r2_score(model.predict(x[i:].values.reshape(-1, 1)), y[i:].values.reshape(-1, 1))
        b = model.intercept_[0]
        a = model.coef_[0][0]
        i += 1
    return a, b, model

