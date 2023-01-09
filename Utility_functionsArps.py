import numpy as np
import math
from scipy.optimize import minimize, Bounds, NonlinearConstraint


class FunctionFluidProduction:
    """Функция добычи жидкости"""

    def __init__(self, day_fluid_production, considerations):
        self.day_fluid_production = day_fluid_production
        self.considerations = considerations
        self.first_m = -1
        self.start_q = -1
        self.ind_max = -1

    def Adaptation(self, correlation_coeff):
        """
        :param correlation_coeff: коэффициенты корреляции функции
        :return: сумма квадратов отклонений фактических точек от модели
        """
        k1, k2 = correlation_coeff
        if self.day_fluid_production.size > 6:
            max_day_prod = np.amax(self.day_fluid_production[:7])
            index = list(np.where(self.day_fluid_production == np.amax(self.day_fluid_production[0:7])))[0][0]
        else:
            max_day_prod = np.amax(self.day_fluid_production)
            index = list(np.where(self.day_fluid_production == np.amax(self.day_fluid_production)))[0][0]

        indexes = np.arange(start=index, stop=self.day_fluid_production.size, step=1) - index
        day_fluid_production_month = max_day_prod * (1 + k1 * k2 * indexes) ** (-1 / k2)
        deviation = [(self.day_fluid_production[index:] - day_fluid_production_month) ** 2]
        self.first_m = self.day_fluid_production.size - index + 1
        self.start_q = max_day_prod
        self.ind_max = index
        return np.sum(deviation)

    def Conditions_FP(self, correlation_coeff):
        """Привязка (binding) к последней точке 2 года"""
        k1, k2 = correlation_coeff
        global base_correction
        point = 1
        if math.isnan(point):
            point = 1
        if point == 1:
            base_correction = self.day_fluid_production[-1]
        elif point == 3:
            if self.day_fluid_production.size >= 3:
                base_correction = np.average(self.day_fluid_production[-3:-1])
            elif self.day_fluid_production.size == 2:
                base_correction = np.average(self.day_fluid_production[-2:-1])
            else:
                base_correction = self.day_fluid_production[-1]
        else:
            print('Неверный формат для условия привязки! Привязка будет осуществлятся к последней точке.')
            base_correction = self.day_fluid_production[-1]

        if self.day_fluid_production.size > 6:
            max_day_prod = np.amax(self.day_fluid_production[:7])
            index = list(np.where(self.day_fluid_production == np.amax(self.day_fluid_production[0:7])))[0][0]
        else:
            max_day_prod = np.amax(self.day_fluid_production)
            index = list(np.where(self.day_fluid_production == np.amax(self.day_fluid_production)))[0][0]

        last_prod = max_day_prod * (1 + k1 * k2 * (self.day_fluid_production.size - 1 - index)) ** (-1 / k2)
        binding = base_correction - last_prod
        return binding


def Calc_FFP(array_production, array_timeProduction, conf):
    """
    Функция для аппроксимации характеристики вытеснения
    :param array_production: массив добычи нефти
    :param array_timeProduction: массив времени работы скважины
    :param conf: ограничения на аппроксимацию
    :return: output - массив с коэффициентами аппроксимации
    [k1, k2, first_m, start_q, index, Qnef_nak]
     0  1       2        3       4       5
    """
    Qnef_nak = np.sum(array_production) / 1000
    array_rates = array_production / (array_timeProduction / 24)
    array_rates[array_rates == -np.inf] = 0
    array_rates[array_rates == np.inf] = 0

    """ Условие, если в расчете только одна точка или последняя точка максимальная """
    if (array_production.size == 1) or (np.amax(array_rates) == array_rates[-1]):
        index = list(np.where(array_rates == np.amax(array_rates)))[0][0]
        first_m = array_rates.size - index + 1
        start_q = array_rates[-1]
        k1 = "Средний темп"
        k2 = "Средний темп"
        output = [k1, k2, first_m, start_q, index, Qnef_nak]
    else:
        # Ограничения:
        k1_left = 0.0001
        k2_left = 0.0001
        k1_right = 1.1
        k2_right = 50

        k1 = 0.0001
        k2 = 0.0001
        c_cet = [k1, k2]
        FP = FunctionFluidProduction(array_rates, conf)
        bnds = Bounds([k1_left, k2_left], [k1_right, k2_right])
        try:
            for i in range(10):
                non_linear_con = NonlinearConstraint(FP.Conditions_FP, [-0.00001], [0.00001])
                res = minimize(FP.Adaptation, c_cet, method='trust-constr', bounds=bnds,
                               constraints=non_linear_con, options={'disp': False})
                c_cet = res.x
                if res.nit < 900:
                    break
            output = [res.x[0],res.x[1], FP.first_m, FP.start_q, FP.ind_max, Qnef_nak]
        except:
            index = list(np.where(array_rates == np.amax(array_rates)))[0][0]
            first_m = array_rates.size - index + 1
            start_q = array_rates[-1]
            output = ["Невозможно", "Невозможно", first_m, start_q, index, Qnef_nak]
    return output
