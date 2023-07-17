import pandas as pd
import numpy as np
from FirstRowWells import first_row_of_well_drainage_front, first_row_of_well_geometry
import geopandas as gpd
from shapely import Point, LineString


def cell_definition(slice_well, df_Coordinates, reservoir_reaction_distance,
                    dict_HHT, df_drainage_areas, wellNumberInj, drainage_areas, **kwargs):
    """
    Опредление ячейки: оружения для каждой нагнетательной скважины
    :param slice_well: исходная таблица МЭР для нагнетательноый скважины
    :param df_Coordinates: массив с координататми для все скважин
    :param reservoir_reaction_distance: словарь максимальных расстояний реагирования для объекта
    :param dict_HHT: словарь нефтенасыщеных толщин скважин
    :param df_drainage_areas: для расчета окружения с зонами дреннирования - массив зон для каждой скважины
    :param wellNumberInj: название нагнетательной скважины
    :param drainage_areas: переключатель для расчета с зонами дреннирования
    :param kwargs: max_overlap_percent, default_distance, angle_verWell, angle_horWell_T1, angle_horWell_T3,
                   DEFAULT_HHT, PROD_MARKER
    :return: df_OneInjCell
    """
    df_OneInjCell = pd.DataFrame()

    # parameters of inj well
    workHorizonInj = df_Coordinates.loc[df_Coordinates.wellNumberColumn == wellNumberInj].workHorizon.iloc[0]
    list_workHorizonInj = workHorizonInj.replace(" ", "").split(",")

    # Расчет расстояния для поиска окружения *выполнен учет работы на несколько пластов
    list_max_distance = []
    for object_inj_well in list_workHorizonInj:
        maximum_distance = reservoir_reaction_distance.get(object_inj_well, [kwargs["default_distance"]])[0]
        list_max_distance.append(maximum_distance)
    maximum_distance = np.mean(list_max_distance)

    H_inj_well = float(dict_HHT.get(wellNumberInj, {"HHT": kwargs["DEFAULT_HHT"]})["HHT"])

    start_date_inj = slice_well.nameDate.iloc[0]
    cumulative_six_month = slice_well.waterInjection.iloc[-6:].sum()

    # wells producing one layer == injection well horizon
    df_WellOneHorizon = df_Coordinates
    df_WellOneHorizon = df_WellOneHorizon[
        list(map(lambda x: len(set(x.replace(" ", "").split(",")) & set(list_workHorizonInj)) > 0,
                 df_WellOneHorizon.workHorizon))]
    if df_WellOneHorizon.shape[0] > 1:
        df_WellOneHorizon = df_WellOneHorizon[['wellNumberColumn',
                                               'workMarker',
                                               'coordinateXT1',
                                               'coordinateYT1',
                                               'coordinateXT3',
                                               'coordinateYT3',
                                               "well type",
                                               "length of well T1-3"]].set_index("wellNumberColumn")
        if drainage_areas:
            # select wells in the injection zone of inj well
            df_WellOneHorizon = df_WellOneHorizon.merge(
                df_drainage_areas[['wellNumberColumn', "drainage_area"]],
                left_on='wellNumberColumn', right_on='wellNumberColumn') \
                .set_index("wellNumberColumn")
            gdf_WellOneHorizon = gpd.GeoDataFrame(df_WellOneHorizon, geometry="drainage_area")
            area_inj = gdf_WellOneHorizon["drainage_area"].loc[wellNumberInj]
            gdf_WellOneArea = gdf_WellOneHorizon[gdf_WellOneHorizon["drainage_area"].intersects(area_inj)]

            # select first row of wells based on drainage areas
            list_first_row_wells = first_row_of_well_drainage_front(gdf_WellOneArea, wellNumberInj)
        else:
            # add shapely types for well coordinates
            df_WellOneHorizon["POINT T1"] = list(map(lambda x, y: Point(x, y),
                                                     df_WellOneHorizon.coordinateXT1,
                                                     df_WellOneHorizon.coordinateYT1))
            df_WellOneHorizon["POINT T3"] = list(map(lambda x, y: Point(x, y),
                                                     df_WellOneHorizon.coordinateXT3,
                                                     df_WellOneHorizon.coordinateYT3))
            df_WellOneHorizon["LINESTRING"] = list(map(lambda x, y: LineString([x, y]),
                                                       df_WellOneHorizon["POINT T1"],
                                                       df_WellOneHorizon["POINT T3"]))

            gdf_WellOneHorizon = gpd.GeoDataFrame(df_WellOneHorizon, geometry="LINESTRING")
            line_inj = gdf_WellOneHorizon["LINESTRING"].loc[wellNumberInj]
            gdf_WellOneHorizon['distance'] = gdf_WellOneHorizon["LINESTRING"].distance(line_inj)

            # select wells in the injection zone (distance < maximumDistance)
            gdf_WellOneArea = gdf_WellOneHorizon[(gdf_WellOneHorizon['distance'] < maximum_distance)]
            df_WellOneArea = pd.DataFrame(gdf_WellOneArea)
            # select first row of wells based on geometry
            list_first_row_wells = first_row_of_well_geometry(df_WellOneArea, wellNumberInj,
                                                              kwargs["angle_verWell"], kwargs["max_overlap_percent"],
                                                              kwargs["angle_horWell_T1"], kwargs["angle_horWell_T3"])

        df_WellOneArea = df_WellOneArea[df_WellOneArea.index.isin(list_first_row_wells)]
        df_WellOneArea = df_WellOneArea.loc[df_WellOneArea.workMarker == kwargs["PROD_MARKER"]]
        df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="№ добывающей", value=df_WellOneArea.index)
        df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Ячейка", value=wellNumberInj)
        df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Объект", value=workHorizonInj)
        df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Дата запуска ячейки", value=start_date_inj)
        df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Расстояние, м",
                             value=df_WellOneArea['distance'].values)
        df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Нн, м", value=H_inj_well)
        df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Закачка за 6 мес, м3",
                             value=cumulative_six_month)

        if df_OneInjCell.empty:
            1 # print("нет окружения")

        df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Нд, м", value=df_OneInjCell["№ добывающей"]
                             .apply(lambda x: float(dict_HHT.get(x, {"HHT": kwargs["DEFAULT_HHT"]})["HHT"])))
    else:
        1 # print("нет окружения")

    return df_OneInjCell


def calculation_coefficients(df_injCelles, initial_coefficient):
    """
    Расчет коэффициентов участия и влияния
    :param df_injCelles: Исходный массив
    :param initial_coefficient: массив с табличными понижающими коэффициентами
    :return: отредактированный df_injCelles
    """
    # calculation coefficients
    df_injCelles["Расстояние, м"] = df_injCelles["Расстояние, м"].where(df_injCelles["Расстояние, м"] != 0, 100)
    df_injCelles["Кнаг"] = df_injCelles["Нд, м"] / df_injCelles["Расстояние, м"]
    df_injCelles["Кдоб"] = df_injCelles["Нн, м"] / df_injCelles["Расстояние, м"]
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
    initial_coefficient.columns = ["Куч доб табл", "Расстояние, м"]
    df_coeff = initial_coefficient.astype('float64')

    df_injCelles = df_injCelles.sort_values(by="Расстояние, м").reset_index(drop=True)
    df_merge = pd.merge_asof(df_injCelles["Расстояние, м"],
                             df_coeff.sort_values(by="Расстояние, м"), on="Расстояние, м", direction="nearest")
    df_injCelles["Куч доб Итог"] = df_injCelles["Куч доб"].where(df_injCelles["Куч доб"] != 1, df_merge["Куч доб табл"])

    df_injCelles["Закачка за 6 мес, м3"] = df_injCelles["Закачка за 6 мес, м3"] * df_injCelles["Куч доб"]
    sum_cumulative = df_injCelles[["№ добывающей", "Закачка за 6 мес, м3"]].groupby(by=["№ добывающей"]).sum()

    df_injCelles["Дин Куч доб"] = df_injCelles.apply(lambda row: df_injCelles["Закачка за 6 мес, м3"].iloc[row.name] /
                                                     sum_cumulative.loc[df_injCelles["№ добывающей"].iloc[row.name]],
                                                     axis=1)

    df_injCelles["Дин Куч доб Итог"] = df_injCelles["Дин Куч доб"].where(df_injCelles["Дин Куч доб"] != 1,
                                                                         df_merge["Куч доб табл"])
    return df_injCelles

