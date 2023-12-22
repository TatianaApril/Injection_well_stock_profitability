import pandas as pd
from FirstRowWells import first_row_of_well_drainage_front, first_row_of_well_geometry
import geopandas as gpd
from shapely import Point, LineString

from Schema import sample_df_injCells_horizon
from Spearman_rho import Spearman_rho_calculation
from tqdm import tqdm


def cell_definition(slice_well, df_Coordinates, df_prod_horizon, reservoir_reaction_distance,
                    dict_HHT, df_drainage_areas, wellNumberInj, drainage_areas, **kwargs):
    """
    Опредление ячейки: оружения для каждой нагнетательной скважины
    :param df_prod_horizon: таблица с динамикой добычи для доб скв
    :param slice_well: исходная таблица МЭР для нагнетательной скважины
    :param df_Coordinates: массив с координататми для всех скважин
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
    object_inj_well = slice_well.Horizon.iloc[-1]
    cluster_inj = df_Coordinates[df_Coordinates.Well_number == wellNumberInj]["Well_cluster"].iloc[-1]
    maximum_distance = reservoir_reaction_distance.get(object_inj_well, [kwargs["default_distance"]])[0]
    H_inj_well = float(dict_HHT.get(wellNumberInj, {"HHT": kwargs["DEFAULT_HHT"]})["HHT"])
    start_date_inj = slice_well.Date.iloc[0]  # .min()?

    # Параметр для динамического Куч
    if slice_well.shape[0] >= 3:
        injection_mean = slice_well.Injection_rate.iloc[-3:].mean()
    else:
        injection_mean = slice_well.Injection_rate.iloc[-1:].mean()

    # wells producing one layer == injection well horizon
    df_Coordinates = df_Coordinates.set_index("Well_number")
    if drainage_areas:
        # select wells in the injection zone of inj well
        df_Coordinates = df_Coordinates.merge(df_drainage_areas[['Well_number', "drainage_area"]],
                                              left_on='Well_number', right_on='Well_number').set_index("Well_number")
        gdf_Coordinates = gpd.GeoDataFrame(df_Coordinates, geometry="drainage_area")
        area_inj = gdf_Coordinates["drainage_area"].loc[wellNumberInj]
        gdf_WellOneArea = gdf_Coordinates[gdf_Coordinates["drainage_area"].intersects(area_inj)]

        # select first row of wells based on drainage areas
        list_first_row_wells = first_row_of_well_drainage_front(gdf_WellOneArea, wellNumberInj)
    else:
        # add shapely types for well coordinates
        df_Coordinates["POINT T1"] = list(map(lambda x, y: Point(x, y), df_Coordinates.XT1, df_Coordinates.YT1))
        df_Coordinates["POINT T3"] = list(map(lambda x, y: Point(x, y), df_Coordinates.XT3, df_Coordinates.YT3))
        df_Coordinates["LINESTRING"] = list(map(lambda x, y: LineString([x, y]),
                                                df_Coordinates["POINT T1"],
                                                df_Coordinates["POINT T3"]))

        gdf_Coordinates = gpd.GeoDataFrame(df_Coordinates, geometry="LINESTRING")
        line_inj = gdf_Coordinates["LINESTRING"].loc[wellNumberInj]
        gdf_Coordinates['distance'] = gdf_Coordinates["LINESTRING"].distance(line_inj)

        # select wells in the injection zone (distance < maximumDistance)
        gdf_OneArea = gdf_Coordinates[(gdf_Coordinates['distance'] < maximum_distance)]
        df_OneArea = pd.DataFrame(gdf_OneArea)
        # select first row of wells based on geometry
        list_first_row_wells = first_row_of_well_geometry(df_OneArea, wellNumberInj,
                                                          kwargs["angle_verWell"], kwargs["max_overlap_percent"],
                                                          kwargs["angle_horWell_T1"], kwargs["angle_horWell_T3"])

    df_OneArea = df_OneArea[df_OneArea.index.isin(list_first_row_wells)]
    df_OneArea = df_OneArea.loc[df_OneArea["well marker"] == "prod"]
    # Параметр для динамического Квл
    df_prod_horizon = df_prod_horizon[df_prod_horizon.Well_number.isin(df_OneArea.index)]
    series_mean_rate = pd.Series()
    for i in df_prod_horizon.Well_number.unique():
        if slice_well.shape[0] >= 3:
            rate_mean = df_prod_horizon[df_prod_horizon.Well_number == i]['Rate_fluid'].iloc[-3:].mean()
        else:
            rate_mean = df_prod_horizon[df_prod_horizon.Well_number == i]['Rate_fluid'].iloc[-1:].mean()
        series_mean_rate.at[i] = rate_mean

    df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="№ добывающей", value=df_OneArea.index)
    df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Куст добывающей", value=df_OneArea.Well_cluster.values)
    df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Ячейка", value=wellNumberInj)
    df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Куст нагнетательной", value=cluster_inj)
    df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Объект", value=object_inj_well)
    df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Дата запуска ячейки", value=start_date_inj)
    df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Расстояние, м",
                         value=df_OneArea['distance'].values)
    df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Нн, м", value=H_inj_well)
    df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Средняя закачка за 3 мес, м3",
                         value=injection_mean)
    df_OneInjCell = df_OneInjCell.merge(series_mean_rate.rename('Средний дебит за 3 мес, м3'),
                                        left_on='№ добывающей', right_index=True)
    df_OneInjCell.insert(loc=df_OneInjCell.shape[1], column="Нд, м", value=df_OneInjCell["№ добывающей"]
                         .apply(lambda x: float(dict_HHT.get(x, {"HHT": kwargs["DEFAULT_HHT"]})["HHT"])))

    # Расчет коэффицента Спирмена
    df_Spearman = pd.DataFrame()
    for i in df_prod_horizon.Well_number.unique():
        row_Spearman = Spearman_rho_calculation(slice_well, df_prod_horizon[df_prod_horizon.Well_number == i],
                                                df_OneArea.loc[i]['distance'])
        df_Spearman = df_Spearman.append(row_Spearman, ignore_index=True)
    if not df_Spearman.empty:
        df_OneInjCell = pd.merge(df_OneInjCell, df_Spearman, how='left', left_on=["№ добывающей", "Ячейка"],
                                 right_on=["№ Добывающей скважины", "№ Нагнетательной скважины"])
        del df_OneInjCell["№ Добывающей скважины"]
        del df_OneInjCell["№ Нагнетательной скважины"]
    return df_OneInjCell


def calculation_coefficients(df_injCelles, initial_coefficient, dynamic_coefficient):
    """
    Расчет коэффициентов участия и влияния
    :param dynamic_coefficient: bool расчет динамического или геометрического коэффициенитов
    :param df_injCelles: Исходный массив
    :param initial_coefficient: массив с табличными понижающими коэффициентами
    :return: отредактированный df_injCelles
    """
    # calculation coefficients
    df_injCelles["Расстояние, м"] = df_injCelles["Расстояние, м"].where(df_injCelles["Расстояние, м"] != 0, 100)
    if dynamic_coefficient:
        df_injCelles["Кнаг"] = df_injCelles["Нд, м"] * df_injCelles['Средний дебит за 3 мес, м3'] \
                               / df_injCelles["Расстояние, м"]
        df_injCelles["Кдоб"] = df_injCelles["Нн, м"] * df_injCelles["Средняя закачка за 3 мес, м3"] \
                               / df_injCelles["Расстояние, м"]
    else:
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
    df_injCelles["Куч Итог"] = df_injCelles["Куч доб"].where(df_injCelles["Куч доб"] != 1, df_merge["Куч доб табл"])
    return df_injCelles


def calculation_injCelle(list_inj_wells, df_Coordinates_horizon, df_inj_horizon, df_prod_horizon,
                         reservoir_reaction_distance, dict_HHT, df_drainage_areas, drainage_areas, **kwargs):
    """
    Разбивка всего фонда скважин на ячейки
    :param df_inj_horizon: таблица с динамикой добычи для наг скв
    :param drainage_areas: переключатель для расчета с зонами дреннирования
    :param list_inj_wells: список нагнетательных скважин
    :param df_Coordinates_horizon: массив с координататми для всех скважин
    :param df_prod_horizon: таблица с динамикой добычи для доб скв
    :param reservoir_reaction_distance: словарь максимальных расстояний реагирования для объекта
    :param dict_HHT: словарь нефтенасыщеных толщин скважин
    :param df_drainage_areas: для расчета окружения с зонами дреннирования - массив зон для каждой скважины
    :param kwargs: max_overlap_percent, default_distance, angle_verWell, angle_horWell_T1, angle_horWell_T3,
                   DEFAULT_HHT
    :return: df_OneInjCell
    """
    # sample of Dataframe: injCelles_horizon
    df_injCells_horizon = sample_df_injCells_horizon
    for inj_well in tqdm(list_inj_wells, desc='I. calculation of cells'):
        slice_well = df_inj_horizon.loc[df_inj_horizon.Well_number == inj_well]

        df_OneInjCelle = cell_definition(slice_well, df_Coordinates_horizon, df_prod_horizon,
                                         reservoir_reaction_distance, dict_HHT, df_drainage_areas,
                                         inj_well, drainage_areas, **kwargs)
        df_injCells_horizon = df_injCells_horizon.append(df_OneInjCelle, ignore_index=True)
    df_injCells_horizon["Ячейка"] = df_injCells_horizon["Ячейка"].astype("str")
    df_injCells_horizon = df_injCells_horizon.sort_values(by="Ячейка").reset_index(drop=True)

    return df_injCells_horizon