import math


def get_properties(dict_properties, *list_horizons):
    """
    get properties from the dictionary; return averaged properties if there are several horizons
    :param dict_properties: dictionary from reservoir_properties.yml
    :param list_horizons: horizons on which the well works
    :return:
    """
    Bo, Bw, Ro, m, So, So_min = [], [], [], [], []
    for horizon in list_horizons:
        pass
    return


def R_prod(cumulative_oil_prod, Bo, Ro, H, m, So, So_min, type_well, len_well):
    """
    calculate drainage radius of production well
    :param cumulative_oil_prod: cumulative oil production, tons
    :param Bo: volumetric ratio of oil
    :param Ro: oil density g/cm3
    :param H: effective thickness, m
    :param m: reservoir porosity, units
    :param So: initial oil saturation, units
    :param So_min: minimum oil saturation, units
    :param type_well: "vertical" or "horizontal"
    :param len_well: length of well for vertical well
    :return:
    """
    if type_well == "vertical":
        a = cumulative_oil_prod * Bo
        b = Ro * math.pi * H * m * (So - So_min)
        R = math.sqrt(a / b)
        return R
    elif type_well == "horizontal":
        L = len_well
        a = math.pi * cumulative_oil_prod * Bo
        b = H * m * Ro * (So - So_min)
        R = (-1 * L + math.sqrt(L * L + a / b)) / math.pi
        return R
    else:
        raise NameError(f"Wrong well type: {type_well}. Allowed values: vertical or horizontal")


def R_inj(cumulative_water_inj, Bw, H, m, So, So_min, type_well, len_well):
    """
    calculate drainage radius of injection well
    :param cumulative_water_inj: накопленная закачка воды, м3
    :param Bw: volumetric ratio of water
    :param H: effective thickness, m
    :param m: reservoir porosity, units
    :param So: initial oil saturation, units
    :param So_min: minimum oil saturation, units
    :param type_well: "vertical" or "horizontal"
    :param len_well: length of well for vertical well
    :return:
    """
    if type_well == "vertical":
        R = math.sqrt(cumulative_water_inj * Bw / (math.pi * H * m * (So - So_min)))
        return R
    elif type_well == "horizontal":
        L = len_well
        a = math.pi * cumulative_water_inj
        b = H * m * (So - So_min)
        R = (-1 * L + math.sqrt(L * L + a / b)) / math.pi
        return R
    else:
        raise NameError(f"Wrong well type: {type_well}. Allowed values: vertical or horizontal")