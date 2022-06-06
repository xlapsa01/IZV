#!/usr/bin/python3.8
# coding=utf-8
from matplotlib.colors import Colormap
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import shape
import datetime
import sklearn.cluster
import os
import numpy as np
# muzete pridat vlastni knihovny


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.
    GeoDataFrame se spravnym kodovani
    """

    df = df.dropna(subset=["d", "e"])

    return geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df["d"], df["e"]),
        crs="EPSG:5514")


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s sesti podgrafy podle lokality nehody
     (dalnice vs prvni trida) pro roky 2018-2020 """

    gdf = gdf.astype({"p2a": "datetime64", "p36": "int8"})
    gdf = gdf[(gdf.p2a >= datetime.datetime(2018, 1, 1))]
    gdf = gdf[(gdf.p36 <= 1)]
    gdf_JHM = gdf[gdf.region == "JHM"]
    gdf_JHM = gdf_JHM.to_crs("epsg:3857")
    gdf_JHM_2020 = gdf_JHM[gdf_JHM.p2a >= datetime.datetime(2020, 1, 1)]
    gdf_JHM_2019 = gdf_JHM[
        (gdf_JHM.p2a >= datetime.datetime(2019, 1, 1)) &
        (gdf_JHM.p2a < datetime.datetime(2020, 1, 1))]
    gdf_JHM_2018 = gdf_JHM[gdf_JHM.p2a < datetime.datetime(2019, 1, 1)]

    years = [gdf_JHM_2020, gdf_JHM_2019, gdf_JHM_2018]

    title_num = 0

    titles = ["Nehody v JHM kraji za rok 2020: na dialnici",
              "Nehody v JHM kraji za rok 2020: na ceste 1. triedy",
              "Nehody v JHM kraji za rok 2019: na dialnici",
              "Nehody v JHM kraji za rok 2019: na ceste 1. triedy",
              "Nehody v JHM kraji za rok 2018: na dialnici",
              "Nehody v JHM kraji za rok 2018: na ceste 1. triedy"]

    colors = ["green", "purple"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    for i, axe in enumerate(axes):
        for num, x in enumerate(axe):
            years[i][years[i].p36 == num].plot(
                ax=x,
                markersize=1,
                color=colors[num])
            x.set_title(titles[title_num], pad=10)
            x.axis("off")
            title_num += 1
            ctx.add_basemap(
                x, crs=gdf_JHM.crs.to_string(),
                source=ctx.providers.Stamen.TonerLite)

    plt.tight_layout()

    if fig_location is not None:
        index = fig_location[::-1].find('/')
        if index == -1:
            file_name = fig_location
        else:
            file_name = fig_location[-index:]
            os.makedirs(fig_location[:-index], exist_ok=True)
        plt.savefig(fig_location)

    if show_figure:
        plt.show()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru"""

    region = "KVK"
    gdf = gdf.astype({"p36": "int8"})
    gdf_reg = gdf[(gdf.region == "{}".format(region)) & (gdf.p36 == 1)]

    gdf_reg = gdf_reg.set_geometry(gdf_reg.centroid).to_crs(epsg=3857)
    gdf_reg = gdf_reg[~gdf_reg.geometry.is_empty]

    coords = np.dstack([gdf_reg.geometry.x, gdf_reg.geometry.y]).reshape(-1, 2)
    db = sklearn.cluster.MiniBatchKMeans(n_clusters=10).fit(coords)

    gdf4 = gdf_reg.copy()
    gdf4["cluster"] = db.labels_
    gdf5 = gdf4.dissolve(by="cluster", aggfunc={"p1": "count"})

    plt.figure(figsize=(9, 11))
    ax = plt.gca()

    gdf5.plot(ax=ax, markersize=0.8,
              column="p1",
              legend=True,
              legend_kwds={'orientation': "horizontal", 'shrink': 0.7})

    ctx.add_basemap(ax, crs="epsg:3857", source=ctx.providers.Stamen.TonerLite)
    ax.set_title("Nehody v {} kraji: na ceste 1. triedy".format(region),
                 pad=20)
    ax.set_aspect("auto")
    ax.axis("off")

    if fig_location is not None:
        index = fig_location[::-1].find('/')
        if index == -1:
            file_name = fig_location
        else:
            file_name = fig_location[-index:]
            os.makedirs(fig_location[:-index], exist_ok=True)
        plt.savefig(fig_location)

    if show_figure:
        plt.show()


if __name__ == "__main__":
    pass
