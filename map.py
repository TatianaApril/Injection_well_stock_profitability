import pandas as pd
import os
import geopandas as gdp
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, polygon
import numpy as np


def true_polygon(x, y):
    """
    Порядок точек для обхода периметра многоугольника по часовой стрелке
    :param x: список х-координат точек
    :param y: список y - координат
    :return: новый порядок индексов
    """
    def rotate(A, B, C):
        return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])

    points = list(zip(x, y))
    indices = list(range(len(x)))
    for i in range(2, len(x)):
        j = i
        while j > 1 and (rotate(points[indices[0]], points[indices[j - 1]], points[indices[j]]) < 0):
            indices[j], indices[j - 1] = indices[j - 1], indices[j]
            j -= 1
    return indices


def make_map_cells(df_cells, df_coordinates, list_names, name):
    """
    Создание карты ячеек
    :param name: название файла
    :param df_cells: лист с ячейками
    :param df_coordinates: лист с координатами
    :param list_names: перечень имен столбцов в таблице
    """
    # Parameters

    name_wellColumn, nameXT1, nameYT1, nameXT3, nameYT3, workMarker, prodMarker, injMarker = list_names
    fontsize = 3  # Размер шрифта
    linewidth = 0.6  # Толщина линий в точках
    size_point = (max(df_coordinates[nameXT1]) - min(df_coordinates[nameXT1])) * 0.0001  # Размер точки

    # Добавление типов shapely для координат скважин
    df_coordinates["POINT T1"] = list(map(lambda x, y: Point(x, y),
                                          df_coordinates[nameXT1],
                                          df_coordinates[nameYT1]))
    df_coordinates["POINT T3"] = list(map(lambda x, y: Point(x, y),
                                          df_coordinates[nameXT3],
                                          df_coordinates[nameYT3]))
    df_coordinates["LINESTRING"] = list(map(lambda x, y: LineString([x, y]),
                                            df_coordinates["POINT T1"],
                                            df_coordinates["POINT T3"]))

    listWellsInj = list(df_cells["Ячейка"].unique())  # Список актуальных ячеек
    geo_cells = pd.DataFrame(columns=["Ячейки", "№ добывающих", "POLYGON"])

    #  Цикл по созданию полигона, который будет отрисован для каждой ячейки
    for wellNumberInj in listWellsInj:
        print(f"{str(wellNumberInj)} {str(int(100 * (listWellsInj.index(wellNumberInj) + 1) / len(listWellsInj)))}%")
        number_prodWell = df_cells[df_cells["Ячейка"] == wellNumberInj]["№ добывающей"].values
        x = df_coordinates[df_coordinates[name_wellColumn].isin(number_prodWell)][nameXT1].values
        y = df_coordinates[df_coordinates[name_wellColumn].isin(number_prodWell)][nameYT1].values
        x_inj = df_coordinates[df_coordinates[name_wellColumn] == wellNumberInj][nameXT1].values[0]
        y_inj = df_coordinates[df_coordinates[name_wellColumn] == wellNumberInj][nameYT1].values[0]
        try:
            Polygon(zip(x, y))
        except ValueError:
            x = np.append(x, x_inj)
            y = np.append(y, y_inj)
            if len(x) == 2:
                L = 100
                x = np.append(x, x[0] + L)
                y = np.append(y, y[0])
        polygon_cell = Polygon(list(map(lambda i: list(zip(x, y))[i], true_polygon(x, y))))
        if not polygon_cell.contains(Point(x_inj, y_inj)):
            x = np.append(x, x_inj)
            y = np.append(y, y_inj)
            polygon_cell = Polygon(list(map(lambda i: list(zip(x, y))[i], true_polygon(x, y))))
        geo_cells = geo_cells.append({'Ячейки': wellNumberInj,
                                      '№ добывающих': number_prodWell,
                                      'POLYGON': polygon_cell},
                                     ignore_index=True)

    # Активация геометрии для массива ячеек
    geo_cells = gdp.GeoDataFrame(geo_cells, geometry=geo_cells['POLYGON'])

    # Активация геометрии для массива координат
    geo_coords = gdp.GeoDataFrame(df_coordinates, geometry=df_coordinates["LINESTRING"])
    gdf = geo_cells.set_index('Ячейки')
    gdf["area"] = gdf.area

    # Вычисление характерной точки, которая точно лежит в пределах полигона ячейки
    gdf['coords_polygon'] = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
    gdf['coords_polygon'] = [coords[0] for coords in gdf['coords_polygon']]

    #  Создание карты и областей полигонов ячеек
    ax = gdf.plot("area", cmap='OrRd', figsize=[30, 30])

    # Подпись ячеек
    for idx, row in gdf.iterrows():
        plt.annotate(text=idx, xy=row['coords_polygon'], color="red", horizontalalignment='center', fontsize=fontsize)

    # Отрисовка границ ячеек
    geo_cells = gdp.GeoDataFrame(geo_cells, geometry=geo_cells['POLYGON'])
    geo_cells.boundary.plot(ax=ax, linewidth=linewidth)

    # Горизонтальные стволы скважин
    geo_coords.plot(ax=ax, color="black", linewidth=linewidth)

    # Подпись нагнетательных скважин
    geo_coords = geo_coords.set_geometry(geo_coords["POINT T1"])
    for x, y, label in zip(df_coordinates[df_coordinates[workMarker] == injMarker][nameXT1].values,
                           df_coordinates[df_coordinates[workMarker] == injMarker][nameYT1].values,
                           df_coordinates[df_coordinates[workMarker] == injMarker][name_wellColumn].values):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", color="navy", fontsize=fontsize)

    # Точки скважин - черные добывающие, синие треугольники - нагнетательные
    geo_coords[df_coordinates[workMarker] == injMarker].plot(ax=ax, color="blue", markersize=size_point, marker="^")
    geo_coords[df_coordinates[workMarker] == prodMarker].plot(ax=ax, color="black", markersize=size_point)

    # Сохранение файла
    plt.savefig(f'files/picture_of_{name}.png', dpi=700, quality=100)
    plt.show()
    pass


if __name__ == '__main__':
    data_output = "files/Вынгапур.xlsx"
    name = data_output.replace("files/", "").replace(".xlsx", "")

    wellNumberColumn = '№ скважины'
    coordinateXT1 = "Координата X"
    coordinateYT1 = "Координата Y"
    coordinateXT3 = "Координата забоя Х (по траектории)"
    coordinateYT3 = "Координата забоя Y (по траектории)"
    workMarker = 'Характер работы'
    prodMarker = "НЕФ"
    injMarker = "НАГ"

    list_names = [wellNumberColumn,
                  coordinateXT1,
                  coordinateYT1,
                  coordinateXT3,
                  coordinateYT3,
                  workMarker,
                  prodMarker,
                  injMarker]

    df_cells = pd.read_excel(os.path.join(os.path.dirname(__file__), data_output), sheet_name="Ячейки")
    df_Coordinates = pd.read_excel(os.path.join(os.path.dirname(__file__), data_output), sheet_name="Координаты")

    make_map_cells(df_cells, df_Coordinates, list_names, name)