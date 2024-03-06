import pandas as pd
import numpy as np
import math

from scipy.stats import spearmanr


def Stop_time_check(history, max_delta):
    second_date = history.Date.iloc[1:]
    last_data = history.Date.iloc[-1]
    second_date.loc[-1] = last_data
    history["След. дата"] = list(second_date)
    history["Разница дат"] = history["След. дата"] - history.Date
    """ Обрезка истории если скважина была остановлена дольше max_delta """
    if not history[history["Разница дат"] > np.timedelta64(max_delta, 'D')].empty:
        last_index = history[history["Разница дат"] > np.timedelta64(max_delta, 'D')].index.tolist()[-1]
        history = history.loc[last_index + 1:]
    del history["Разница дат"]
    del history["След. дата"]
    return history


def Spearman_coeff_calculation(name_column, history, injectivity, max_lag_time):
    list_coef = []
    com_flow = history.shape[0]
    for i in range(max_lag_time + 1):
        liq = history[name_column].loc[
              com_flow - injectivity.shape[0] - max_lag_time + i:com_flow - max_lag_time - 1 + i]
        corr, p_value = spearmanr(injectivity, liq)
        list_coef.append(round(corr, 3))
    return list_coef


def Check_connection_quality(s_Qzh, S_Qn, S_Pz):
    """
    Шкала для качестсва связи:
     1) нет гидродинамической связи
     2) низкая связь с возможным потенциалом - для реализации необходимо изменение тех.режима добывающей скважины
     3) очень слабая связь
     4) слабая связь
     5) слабая связь с возможным потенциалом
     6) умеренная связь
     7) умеренная связь с возможным потенциалом
     8) заметная связь
     9) заметная связь с возможным потенциалом
     10) заметная связь с возможным промывом
     11) высокая связь
     12) высокая связь с возможным промывом
     13) очень высокая связь
    """
    if math.isnan(s_Qzh):
        return "невозможно рассчитать качество связи (приемистость на полке)"

    if type(S_Pz) == str:
        S_Pz = -2

    if s_Qzh <= 0.1:
        if (S_Pz == -2) or (S_Pz <= 0.3):
            return "нет гидродинамической связи"
        elif 0.3 < S_Pz <= 1:
            return "низкая связь с возможным потенциалом - " \
                   "для реализации необходимо изменение тех.режима добывающей скважины"

    elif 0.1 < s_Qzh <= 0.2:
        if S_Pz <= 0.1:
            return "очень слабая связь"
        elif 0.1 < S_Pz <= 0.4:
            return "слабая связь"
        else:
            return "слабая связь с возможным потенциалом"

    elif 0.2 < s_Qzh <= 0.3:
        if S_Pz <= 0.4:
            if S_Qn <= 0.2:
                return "слабая связь"
            else:
                return "умеренная связь"
        else:
            if S_Qn <= 0.2:
                return "умеренная связь с возможным потенциалом"
            else:
                return "заметная связь"

    elif 0.3 < s_Qzh <= 0.5:
        if S_Qn <= 0.3:
            if S_Pz <= 0.4:
                return "умеренная связь"
            else:
                return "заметная связь с возможным потенциалом"
        elif 0.3 < S_Qn <= 0.5:
            return "заметная связь"
        else:
            return "высокая связь"

    else:
        if S_Qn <= 0.2:
            if S_Pz <= 0.4:
                return "заметная связь с возможным промывом"
            else:
                return "высокая связь с возможным промывом"
        elif 0.2 < S_Qn <= 0.5:
            return "высокая связь"
        else:
            return "очень высокая связь"


def Spearman_rho_calculation(slice_inj, slice_prod, distance):
    max_delta = 200  # Максимальный период остановки, дни (пока 6 мясяцев) 200
    min_period = 5  # Минимальный период совместный работы
    limiting_radius = 500  # Если расстояние больше 500 м то коэффициент искать с задержкой от 2 месяцев
    df_result = pd.DataFrame(columns=["№ Нагнетательной скважины", "№ Добывающей скважины", "Лаг",
                                      "Коэф. Спирмена Жидкость", "Коэф. Спирмена Нефть", "Коэф. Спирмена Pзаб",
                                      "Качество связи"])

    slice_inj = Stop_time_check(slice_inj, max_delta)
    slice_prod = Stop_time_check(slice_prod, max_delta)

    slice_all = slice_inj.merge(slice_prod, how='inner', on="Date")

    # Проверка совместной истории работы
    com_flow = slice_all.shape[0]
    if com_flow < min_period:
        new_row = {"№ Нагнетательной скважины": slice_inj.Well_number.unique()[0],
                   "№ Добывающей скважины": slice_prod.Well_number.unique()[0], "Лаг": None,
                   "Коэф. Спирмена Жидкость": None, "Коэф. Спирмена Нефть": None, "Коэф. Спирмена Pзаб": None,
                   "Качество связи": "короткая совместная история работы, меньше года"}
    else:
        #  Расчет коэффициентов Спирмена
        if com_flow >= 36:
            max_lag_time = 6
            injectivity = slice_all.Injection_rate.loc[com_flow - max_lag_time - 30:com_flow - max_lag_time - 1]
        elif 36 > com_flow >= 18:
            max_lag_time = 6
            injectivity = slice_all.Injection_rate.loc[0:com_flow - max_lag_time - 1]
        else:
            max_lag_time = 2
            injectivity = slice_all.Injection_rate.loc[0:com_flow - max_lag_time - 1]

        list_coef_Qzh = Spearman_coeff_calculation("Rate_fluid", slice_all, injectivity, max_lag_time)
        list_coef_Qn = Spearman_coeff_calculation("Rate_oil", slice_all, injectivity, max_lag_time)

        # Если менее 70% не нулевых значений Pзаб, то не рассчитывать коэф
        if (slice_all['Pbh_y'] != 0).sum() / slice_all['Pbh_y'].shape[0] > 0.7:
            list_coef_Pz = Spearman_coeff_calculation('Pbh_y', slice_all, injectivity, max_lag_time)
        else:
            list_coef_Pz = 'не достаточно значений для расчета'

        if distance > limiting_radius:
            S_Qzh = max(list_coef_Qzh[2:])
        else:
            S_Qzh = max(list_coef_Qzh)
        S_Qn = list_coef_Qn[list_coef_Qzh.index(S_Qzh)]

        if type(list_coef_Pz) != str:
            S_Pz = max(list_coef_Pz[:list_coef_Qzh.index(S_Qzh) + 1])
        else:
            S_Pz = list_coef_Pz
        new_row = {"№ Нагнетательной скважины":  slice_inj.Well_number.unique()[0],
                   "№ Добывающей скважины": slice_prod.Well_number.unique()[0], "Лаг": list_coef_Qzh.index(S_Qzh),
                   "Коэф. Спирмена Жидкость": S_Qzh, "Коэф. Спирмена Нефть": S_Qn, "Коэф. Спирмена Pзаб": S_Pz,
                   "Качество связи": Check_connection_quality(S_Qzh, S_Qn, S_Pz)}
    df_result = pd.DataFrame([new_row])
    return df_result


