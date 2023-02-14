import math
from statistics import mean
from shapely import Point, unary_union, LineString
import matplotlib.pyplot as plt
import geopandas as gpd


def get_properties(dict_reservoir, list_horizons):
    """
    get properties from the dictionary; return averaged properties if there are several horizons
    :param dict_reservoir: dictionary from reservoir_properties.yml
    :param list_horizons: horizons on which the well works
    :return:
    """
    # Check there are all horizons in dictionary
    difference = set(list_horizons).difference(set(list(dict_reservoir.keys())))
    if len(difference) > 0:
        raise KeyError(f"There is no properties for horizons: {difference}")

    dict_properties = {'Bo': [], 'Bw': [], 'Ro': [], 'm': [], 'So': [], 'So_min': []}
    for horizon in list_horizons:
        dict_horizon = dict_reservoir[horizon]
        [dict_properties[prop].append(dict_horizon[prop]) for prop in dict_properties.keys()]
    [dict_properties.update({prop: round(mean(dict_properties[prop]), 4)}) for prop in dict_properties.keys()]
    return dict_properties


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


def get_polygon_well(R_well, type_well, *coordinates):
    if type_well == "vertical":
        well_polygon = Point(coordinates[0], coordinates[1]).buffer(R_well)
        return well_polygon
    elif type_well == "horizontal":
        t1 = Point(coordinates[0], coordinates[1])
        t3 = Point(coordinates[2], coordinates[3])
        well_polygon = LineString([t1, t3]).buffer(R_well, join_style=1)

        """график зоны
        circles = unary_union([t1.buffer(R_well), t3.buffer(R_well)])
        ax = gpd.GeoSeries(well_polygon).plot(color="pink")
        gpd.GeoSeries(circles).plot(ax=ax)
        plt.gca().axis("equal")
        plt.show()"""
        return well_polygon
    else:
        raise NameError(f"Wrong well type: {type_well}. Allowed values: vertical or horizontal")
