import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


def history_processing(history, PROD_MARKER, INJ_MARKER, time_work_min=5):
    """
    Предварительная обработка МЭР
    """
    last_data = history.nameDate.unique()
    last_data = np.sort(last_data)[-1]
    history = history.fillna(0)  # Заполнение пустых ячеек нулями

    history = history[history.workHorizon != 0]  # Оставляем не нулевые объекты
    unique_wells = history.wellNumberColumn.unique()  # Уникальный список скважин (без нулевых значений)
    history = history.sort_values(['wellNumberColumn', 'nameDate'])

    history_new = pd.DataFrame()
    for well in unique_wells:
        slice_well = history.loc[history.wellNumberColumn == well].copy()
        object_well = slice_well.workHorizon.iloc[-1]
        # Оставляем историю с последнего объекта работы
        slice_well = slice_well[slice_well.workHorizon == object_well].reset_index(drop=True)

        # Произведем обработку в зависимости от типа скважины
        marker_well = slice_well.workMarker.iloc[-1]
        if marker_well == PROD_MARKER:
            slice_well = slice_well[(slice_well.oilProduction != 0) &
                                    (slice_well.fluidProduction != 0) &
                                    (slice_well.timeProduction >= time_work_min * 24)]
        elif marker_well == INJ_MARKER:
            slice_well = slice_well[(slice_well.waterInjection != 0) &
                                    (slice_well.timeInjection >= time_work_min * 24)]
        history_new = history_new.append(slice_well, ignore_index=True)

    # Уникальный список скважин, которые в работе на последнюю дату
    unique_wells = history_new.loc[history_new.nameDate == last_data].wellNumberColumn.unique()
    history_new = history_new[history_new.wellNumberColumn.isin(unique_wells)]  # Выделяем историю только этих скважин

    return history_new


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