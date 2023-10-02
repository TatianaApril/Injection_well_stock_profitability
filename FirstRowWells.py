import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from shapely import LineString, Point
import os


def check_well_intersection(df_intersectionWells, MaxOverlapPercent):
    """
    Проверка пересечений между скважинами в исходном массиве, для исключения перекрываемых скважин второго ряда
    :param df_intersectionWells: массив скважин с исходными данными
    :param MaxOverlapPercent: максимальный процент от общей длины ствола скважины, который может быть скрыт при
                             пересечении, чтобы скважина осталась в первом ряду
    :return: список скважин за первым рядом, которые требуется удалить
    """
    list_dropWell = []
    # df_intersectionWells = df_intersectionWells.sort_values(by=['distance'], ascending=False)
    df_intersectionWells = df_intersectionWells.sort_values(by=['r_center'], ascending=False)

    for i in range(df_intersectionWells.shape[0] - 1):
        well_name = df_intersectionWells.index[i]

        remoteFi_Max = max(df_intersectionWells.iloc[i][["fi_t1_grad", "fi_t3_grad"]])
        remoteFi_Min = min(df_intersectionWells.iloc[i][["fi_t1_grad", "fi_t3_grad"]])
        nearFi_Max = df_intersectionWells.iloc[i + 1:][["fi_t1_grad", "fi_t3_grad"]].values.max()
        nearFi_Min = df_intersectionWells.iloc[i + 1:][["fi_t1_grad", "fi_t3_grad"]].values.min()

        df_pilot = pd.DataFrame({"fi": pd.Series([remoteFi_Max, remoteFi_Min, nearFi_Max, nearFi_Min]),
                                 "marker": pd.Series([2, 2, 1, 1])})
        df_pilot = df_pilot.sort_values(by=["fi"], ascending=False)

        position_code = df_pilot["marker"].to_string(index=False).replace("\n", "")
        if position_code == "1221":
            list_dropWell.append(well_name)
        elif position_code == "1122" or position_code == "2211":
            continue
        else:
            OverlapPercent = (df_pilot["fi"].values[1] - df_pilot["fi"].values[2]) / (remoteFi_Max - remoteFi_Min) * 100
            if OverlapPercent > MaxOverlapPercent:
                list_dropWell.append(well_name)
    return list_dropWell


def first_row_of_well_geometry(df_WellOneArea,
                               wellNumberInj,
                               verticalWellAngle, MaxOverlapPercent, angle_horizontalT1, angle_horizontalT3):
    """
    Поиск для нагнетательной скважины окружения первого ряда на основе геометрии
    :param angle_horizontalT3: sector expansion angle for horizontal well's T1 point
    :param angle_horizontalT1: sector expansion angle for horizontal well's T3 point
    :param df_WellOneArea: массив скважин окружения с расстоянием до нагнетательной < maximumDistance
    :param wellNumberInj: номер рассматриваемой нагнетательной скважины
    :param verticalWellAngle: угол для обозначения зоны вертикальных скважин на карте
    :param MaxOverlapPercent: максимальный процент от общей длины ствола скважины, который может быть скрыт при
                             пересечении, чтобы скважина осталась в первом ряду
    :return: listNamesFisrtRowWells - список скважин первого ряда
    """
    #  injection well type check
    if df_WellOneArea["well type"].loc[wellNumberInj] == "vertical":
        """Выбор центральных точек для оценки первого ряда: 
        если нагенатетльная вертикальная используется только точка T1
        горизонтальная -    Т1, середина ствола и Т3
        """
        list_startingPoints = [[df_WellOneArea.XT1.loc[wellNumberInj]], [df_WellOneArea.YT1.loc[wellNumberInj]]]
    else:
        list_startingPoints = [[df_WellOneArea.XT1.loc[wellNumberInj], (df_WellOneArea.XT1.loc[wellNumberInj] +
                                df_WellOneArea.XT3.loc[wellNumberInj]) / 2, df_WellOneArea.XT3.loc[wellNumberInj]],
                               [df_WellOneArea.YT1.loc[wellNumberInj],
                                (df_WellOneArea.YT1.loc[wellNumberInj] + df_WellOneArea.YT3.loc[wellNumberInj]) / 2,
                                df_WellOneArea.YT3.loc[wellNumberInj]]]

    #  checking the points of inj well
    listNamesFisrtRowWells = []
    for point in range(len(list_startingPoints[0])):
        df_OnePoint = df_WellOneArea
        df_OnePoint = df_OnePoint.drop(index=[wellNumberInj])
        #  centering
        X0 = list_startingPoints[0][point]
        Y0 = list_startingPoints[1][point]
        df_OnePoint["X_T1"] = df_OnePoint.XT1 - X0
        df_OnePoint["X_T3"] = df_OnePoint.XT3 - X0
        df_OnePoint["Y_T1"] = df_OnePoint.YT1 - Y0
        df_OnePoint["Y_T3"] = df_OnePoint.YT3 - Y0

        #  to polar coordinate system
        df_OnePoint["r_t1"] = np.sqrt(np.power(df_OnePoint["X_T1"], 2) + np.power(df_OnePoint["Y_T1"], 2))
        df_OnePoint["r_t3"] = np.sqrt(np.power(df_OnePoint["X_T3"], 2) + np.power(df_OnePoint["Y_T3"], 2))
        #  df_OnePoint["r_t1"] = df_OnePoint["distance"]
        #  df_OnePoint["r_t3"] = df_OnePoint["distance"]
        df_OnePoint["fi_t1"] = np.arctan2(df_OnePoint["Y_T1"], df_OnePoint["X_T1"])  # in degree * 180 / np.pi
        df_OnePoint["fi_t3"] = np.arctan2(df_OnePoint["Y_T3"], df_OnePoint["X_T3"])

        #  editing coordinates for vertical wells
        df_OnePoint["r_t1"] = df_OnePoint["r_t1"].where((df_OnePoint["well type"] != "vertical"),
                                                        df_OnePoint["r_t1"] / math.cos(verticalWellAngle * np.pi / 180))
        df_OnePoint["r_t3"] = df_OnePoint["r_t1"]
        df_OnePoint["fi_t1"] = df_OnePoint["fi_t1"].where((df_OnePoint["well type"] != "vertical") |
                                                          (df_OnePoint['distance'] == 0),
                                                          df_OnePoint["fi_t1"] + verticalWellAngle * np.pi / 180)

        df_OnePoint["fi_t3"] = df_OnePoint["fi_t3"].where((df_OnePoint["well type"] != "vertical") |
                                                          (df_OnePoint['distance'] == 0),
                                                          df_OnePoint["fi_t3"] - verticalWellAngle * np.pi / 180)

        #  editing coordinates for horizontal wells
        df_OnePoint["fi_t1"] = df_OnePoint["fi_t1"].where((df_OnePoint["well type"] != "horizontal") |
                                                          (df_OnePoint["fi_t1"] < df_OnePoint["fi_t3"]) |
                                                          (df_OnePoint['distance'] == 0),
                                                          df_OnePoint["fi_t1"] + angle_horizontalT1)

        df_OnePoint["fi_t1"] = df_OnePoint["fi_t1"].where((df_OnePoint["well type"] != "horizontal") |
                                                          (df_OnePoint["fi_t1"] > df_OnePoint["fi_t3"]) |
                                                          (df_OnePoint['distance'] == 0),
                                                          df_OnePoint["fi_t1"] - angle_horizontalT1)

        df_OnePoint["fi_t3"] = df_OnePoint["fi_t3"].where((df_OnePoint["well type"] != "horizontal") |
                                                          (df_OnePoint["fi_t3"] < df_OnePoint["fi_t1"]) |
                                                          (df_OnePoint['distance'] == 0),
                                                          df_OnePoint["fi_t3"] + angle_horizontalT3)

        df_OnePoint["fi_t3"] = df_OnePoint["fi_t3"].where((df_OnePoint["well type"] != "horizontal") |
                                                          (df_OnePoint["fi_t3"] > df_OnePoint["fi_t1"]) |
                                                          (df_OnePoint['distance'] == 0),
                                                          df_OnePoint["fi_t3"] - angle_horizontalT3)

        """# график перед очисткой
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for i in range(df_OnePoint["fi_t1"].shape[0]):
            ax.plot([df_OnePoint["fi_t1"].iloc[i], df_OnePoint["fi_t3"].iloc[i]], [df_OnePoint["r_t1"].iloc[i],
                                                                                   df_OnePoint["r_t3"].iloc[i]])
            ax.text(df_OnePoint["fi_t1"].iloc[i], df_OnePoint["r_t1"].iloc[i], df_OnePoint.index[i],
                    fontsize="xx-small", c='k')

        ax.set_title("A line plot on a polar axis: before edit for well " + str(wellNumberInj), va='bottom')
        my_path = os.path.dirname(__file__).replace("\\","/")
        plt.show()
        #plt.savefig(my_path + '/pictures/' + str(wellNumberInj) + "_" + str(point) +' before edit.png')"""


        df_OnePoint["fi_t1_grad"] = df_OnePoint["fi_t1"] * 180 / np.pi
        df_OnePoint["fi_t1_grad"] = df_OnePoint["fi_t1_grad"].where(df_OnePoint["fi_t1_grad"] >= 0,
                                                                    df_OnePoint["fi_t1_grad"] + 360)
        df_OnePoint["fi_t3_grad"] = df_OnePoint["fi_t3"] * 180 / np.pi
        df_OnePoint["fi_t3_grad"] = df_OnePoint["fi_t3_grad"].where(df_OnePoint["fi_t3_grad"] >= 0,
                                                                    df_OnePoint["fi_t3_grad"] + 360)

        # check the wells cross the line 0 degree
        df_OnePoint["fi_min"] = df_OnePoint[["fi_t1_grad", "fi_t3_grad"]].min(axis=1)
        df_OnePoint["fi_max"] = df_OnePoint[["fi_t1_grad", "fi_t3_grad"]].max(axis=1)
        df_OnePoint = df_OnePoint.sort_values(by=["fi_min"])
        df_crossLine = df_OnePoint[(df_OnePoint["fi_min"] >= 0) & (df_OnePoint["fi_min"] <= 90)
                                   & (df_OnePoint["fi_max"] >= 270) & (df_OnePoint["fi_max"] <= 360)]
        if not df_crossLine.empty:
            while not df_crossLine.empty:
                angleRotation = 360 - df_crossLine["fi_max"].min() + 1
                df_OnePoint[["fi_t1_grad", "fi_t3_grad"]] = df_OnePoint[["fi_t1_grad", "fi_t3_grad"]] + angleRotation
                df_OnePoint[["fi_t1_grad", "fi_t3_grad"]] = df_OnePoint[["fi_t1_grad", "fi_t3_grad"]].where(
                    df_OnePoint[["fi_t1_grad", "fi_t3_grad"]] < 360, df_OnePoint[["fi_t1_grad", "fi_t3_grad"]] - 360)

                df_OnePoint["fi_min"] = df_OnePoint[["fi_t1_grad", "fi_t3_grad"]].min(axis=1)
                df_OnePoint["fi_max"] = df_OnePoint[["fi_t1_grad", "fi_t3_grad"]].max(axis=1)
                df_crossLine = df_OnePoint[(df_OnePoint["fi_min"] >= 0) & (df_OnePoint["fi_min"] <= 90)
                                           & (df_OnePoint["fi_max"] >= 270) & (df_OnePoint["fi_max"] <= 360)]
                df_OnePoint = df_OnePoint.sort_values(by=["fi_min"])
                # print("пересекает!" + str(wellNumberInj))

        # well sector center
        df_OnePoint["x_fi_t1"] = df_OnePoint["r_t1"] * np.cos(df_OnePoint["fi_t1"])
        df_OnePoint["y_fi_t1"] = df_OnePoint["r_t1"] * np.sin(df_OnePoint["fi_t1"])
        df_OnePoint["x_fi_t3"] = df_OnePoint["r_t3"] * np.cos(df_OnePoint["fi_t3"])
        df_OnePoint["y_fi_t3"] = df_OnePoint["r_t3"] * np.sin(df_OnePoint["fi_t3"])

        df_OnePoint["x_center"] = (df_OnePoint["x_fi_t1"] + df_OnePoint["x_fi_t3"]) / 2
        df_OnePoint["y_center"] = (df_OnePoint["y_fi_t1"] + df_OnePoint["y_fi_t3"]) / 2

        df_OnePoint["r_center"] = np.sqrt(
            np.power(df_OnePoint["x_center"], 2) + np.power(df_OnePoint["y_center"], 2))

        df_OnePoint = df_OnePoint.sort_values(by=['r_center'], ascending=False)
        # check well intersection
        listNamesOnePointWells = list(df_OnePoint.index)
        listNamesClean = listNamesOnePointWells
        for well in listNamesOnePointWells:
            if well in listNamesClean:
                fi_min = min(df_OnePoint.loc[well]["fi_t1_grad"], df_OnePoint.loc[well]["fi_t3_grad"])
                fi_max = max(df_OnePoint.loc[well]["fi_t1_grad"], df_OnePoint.loc[well]["fi_t3_grad"])
                df_intersectionWells = df_OnePoint.loc[((df_OnePoint["fi_t1_grad"] <= fi_max) &
                                                        (df_OnePoint["fi_t1_grad"] >= fi_min)) |
                                                       ((df_OnePoint["fi_t3_grad"] <= fi_max) &
                                                        (df_OnePoint["fi_t3_grad"] >= fi_min))]
                if df_intersectionWells.shape[0] != 1:
                    list_dropWell = check_well_intersection(df_intersectionWells, MaxOverlapPercent)
                    listNamesClean = list(set(listNamesClean) - set(list_dropWell))

        """# график после очистки
        df_OnePoint = df_OnePoint[df_OnePoint.index.isin(listNamesClean)]
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for i in range(df_OnePoint["fi_t1"].shape[0]):
            ax.plot([df_OnePoint["fi_t1"].iloc[i], df_OnePoint["fi_t3"].iloc[i]],
                    [df_OnePoint["r_t1"].iloc[i],
                     df_OnePoint["r_t3"].iloc[i]])
            ax.text(df_OnePoint["fi_t1"].iloc[i], df_OnePoint["r_t1"].iloc[i], df_OnePoint.index[i],
                    fontsize="xx-small", c='k')
        ax.set_title("A line plot on a polar axis: after edit for well " + str(wellNumberInj), va='bottom')
        plt.show()
        #plt.savefig(my_path + '/pictures/' + str(wellNumberInj) + "_" + str(point) + ' after edit.png')"""

        listNamesFisrtRowWells.extend(listNamesClean)
    listNamesFisrtRowWells = list(set(listNamesFisrtRowWells))
    return listNamesFisrtRowWells


def first_row_of_well_drainage_front(gdf_WellOneArea, wellNumberInj):
    """ В разработке
    Функция аналогичная firstRow_of_Wells_geometry - с учетом областей дреннирования и нагнетания скважин
    :return:
    """
    # map for zones:
    gdf_WellOneArea.insert(loc=gdf_WellOneArea.shape[1], column="area",
                           value=gpd.GeoSeries(gdf_WellOneArea.drainage_area).area)
    df_WellOneArea = gdf_WellOneArea.sort_values(by=['area'], ascending=False)
    # add shapely types for well coordinates
    gdf_WellOneArea.insert(loc=gdf_WellOneArea.shape[1], column="POINT T1",
                           value=list(map(lambda x, y: Point(x, y),
                                          gdf_WellOneArea.XT1, gdf_WellOneArea.YT1)))
    gdf_WellOneArea.insert(loc=gdf_WellOneArea.shape[1], column="POINT T3",
                           value=list(map(lambda x, y: Point(x, y),
                                          gdf_WellOneArea.XT3, gdf_WellOneArea.YT3)))
    gdf_WellOneArea.insert(loc=gdf_WellOneArea.shape[1], column="LINESTRING",
                           value=list(map(lambda x, y: LineString([x, y]),
                                          gdf_WellOneArea["POINT T1"], gdf_WellOneArea["POINT T3"])))

    PROD_MARKER: str = "prod"
    INJ_MARKER: str = "inj"
    fontsize = 6  # Размер шрифта
    size_point = 13

    ax = gpd.GeoSeries(df_WellOneArea.drainage_area).plot(cmap="Blues", figsize=[20, 20])
    area_inj = gdf_WellOneArea["drainage_area"].loc[wellNumberInj].boundary
    gpd.GeoSeries(area_inj).plot(ax=ax, color="red")
    gpd.GeoSeries(gdf_WellOneArea["LINESTRING"]).plot(ax=ax, color="black")

    # Подпись нагнетательных скважин
    for x, y, label in zip(gdf_WellOneArea[gdf_WellOneArea["well marker"] == INJ_MARKER].XT1.values,
                           gdf_WellOneArea[gdf_WellOneArea["well marker"] == INJ_MARKER].YT1.values,
                           gdf_WellOneArea[gdf_WellOneArea["well marker"] == INJ_MARKER].index):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", color="red", fontsize=fontsize)

    for x, y, label in zip(gdf_WellOneArea[gdf_WellOneArea["well marker"] == PROD_MARKER].XT1.values,
                           gdf_WellOneArea[gdf_WellOneArea["well marker"] == PROD_MARKER].YT1.values,
                           gdf_WellOneArea[gdf_WellOneArea["well marker"] == PROD_MARKER].index):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", color="navy", fontsize=fontsize)

    # Точки скважин - черные добывающие, синие треугольники - нагнетательные
    gdf_WellOneArea = gdf_WellOneArea.set_geometry(gdf_WellOneArea["POINT T1"])
    gdf_WellOneArea[gdf_WellOneArea["well marker"] == INJ_MARKER].plot(ax=ax, color="blue",
                                                                   markersize=size_point, marker="^")
    gdf_WellOneArea[gdf_WellOneArea["well marker"] == PROD_MARKER].plot(ax=ax, color="black",
                                                                    markersize=size_point)

    my_path = os.path.dirname(__file__).replace("\\", "/")
    plt.savefig(my_path + '/pictures/' + str(wellNumberInj) + ' zones.png', dpi=200, quality=100)
    print(str(wellNumberInj))
    # plt.show()
    return
