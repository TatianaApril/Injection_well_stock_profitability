import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint


class FunctionFluidProduction:
    """Класс кривой добычи жидкости"""

    def __init__(self, day_fluid_production):
        self.day_fluid_production = day_fluid_production
        self.first_m = -1
        self.start_q = -1
        self.ind_max = -1

    def Adaptation(self, correlation_coeff):
        """
        :param correlation_coeff: коэффициенты корреляции функции
        :return: сумма квадратов отклонений фактических точек от модели
        """
        k1, k2 = correlation_coeff
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
        """Привязка (binding) к последней точке"""
        k1, k2 = correlation_coeff
        base_correction = self.day_fluid_production[-1]

        max_day_prod = np.amax(self.day_fluid_production)
        index = list(np.where(self.day_fluid_production == np.amax(self.day_fluid_production)))[0][0]

        last_prod = max_day_prod * (1 + k1 * k2 * (self.day_fluid_production.size - 1 - index)) ** (-1 / k2)
        binding = base_correction - last_prod
        return binding


def Calc_FFP(array_production, array_timeProduction):
    """
    Функция для аппроксимации кривой добычи
    :param array_production: массив добычи нефти
    :param array_timeProduction: массив времени работы скважины
    :return: output - массив с коэффициентами аппроксимации
    [k1, k2, first_m, start_q, index, cumulative_oil]
     0  1       2        3       4       5
    """
    cumulative_oil = np.sum(array_production) / 1000
    array_rates = array_production / (array_timeProduction / 24)
    array_rates[array_rates == -np.inf] = 0
    array_rates[array_rates == np.inf] = 0

    """ Условие, если в расчете только одна точка или последняя точка максимальная """
    if (array_production.size == 1) or (np.amax(array_rates) == array_rates[-1]):
        index = list(np.where(array_rates == np.amax(array_rates)))[0][0]
        first_m = array_rates.size - index + 1
        start_q = array_rates[-1]
        k1 = 0
        k2 = 1
        output = [k1, k2, first_m, start_q, index, cumulative_oil]
    else:
        # Ограничения:
        k1_left = 0.0001
        k2_left = 0.0001
        k1_right = 1.1
        k2_right = 50

        k1 = 0.0001
        k2 = 0.0001
        c_cet = [k1, k2]
        FP = FunctionFluidProduction(array_rates)
        bnds = Bounds([k1_left, k2_left], [k1_right, k2_right])
        non_linear_con = NonlinearConstraint(FP.Conditions_FP, [-0.00001], [0.00001])
        try:
            res = minimize(FP.Adaptation, c_cet, method='trust-constr', bounds=bnds, constraints=non_linear_con,
                           options={'disp': False, 'xtol': 1E-7, 'gtol': 1E-7, 'maxiter': 2000})
            k1, k2 = res.x[0], res.x[1]
            if k1 < 0:
                k1 = 0
            if k2 < 0:
                k2 = 0
            output = [k1, k2, FP.first_m, FP.start_q, FP.ind_max, cumulative_oil]
        except:
            index = list(np.where(array_rates == np.amax(array_rates)))[0][0]
            first_m = array_rates.size - index + 1
            start_q = array_rates[-1]
            output = ["Невозможно", "Невозможно", first_m, start_q, index, cumulative_oil]
    return output
