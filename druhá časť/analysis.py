#!/usr/bin/env python3.9
# coding=utf-8
from matplotlib import pyplot as plt, scale
import pandas as pd
from pandas.core.reshape.reshape import unstack
import seaborn as sns
import numpy as np
import os
import sys
import datetime

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

""" Ukol 1:
načíst soubor nehod, který byl vytvořen z vašich dat. Neznámé integerové hodnoty byly mapovány na -1.

Úkoly:
- vytvořte sloupec date, který bude ve formátu data (berte v potaz pouze datum, tj sloupec p2a)
- vhodné sloupce zmenšete pomocí kategorických datových typů. Měli byste se dostat po 0.5 GB. Neměňte však na kategorický typ region (špatně by se vám pracovalo s figure-level funkcemi)
- implementujte funkci, která vypíše kompletní (hlubkou) velikost všech sloupců v DataFrame v paměti:
orig_size=X MB
new_size=X MB

Poznámka: zobrazujte na 1 desetinné místo (.1f) a počítejte, že 1 MB = 1e6 B. Počí
"""
# data types for DataFrame
dtypes = {
    "p1": "category",
    "p36": "int8",
    "p37": "int8",
    "p2a": "datetime64",
    "weekday(p2a)": "int8",
    "p2b": "int8",
    "p6": "int8",
    "p7": "int8",
    "p8": "int8",
    "p9": "int8",
    "p10": "int8",
    "p11": "int8",
    "p12": "int8",
    "p13a": "int8",
    "p13b": "int8",
    "p13c": "int8",
    "p14": "int8",
    "p15": "int8",
    "p16": "int8",
    "p17": "int8",
    "p18": "int8",
    "p19": "int8",
    "p20": "int8",
    "p21": "int8",
    "p22": "int8",
    "p23": "int8",
    "p24": "int8",
    "p27": "int8",
    "p28": "int8",
    "p34": "int8",
    "p35": "int8",
    "p39": "int8",
    "p44": "int8",
    "p45a": "int8",
    "p47": "int8",
    "p48a": "int8",
    "p49": "int8",
    "p50a": "int8",
    "p50b": "int8",
    "p51": "int8",
    "p52": "int8",
    "p53": "int8",
    "p55a": "int8",
    "p57": "int8",
    "p58": "int8",
    "a": "float16",
    "b": "float16",
    "d": "float16",
    "e": "float16",
    "f": "category",
    "g": "category",
    "h": "category",
    "i": "category",
    "j": "category",
    "k": "category",
    "l": "category",
    "n": "category",
    "o": "category",
    "p": "category",
    "q": "category",
    "r": "int8",
    "s": "int8",
    "t": "category",
    "p5a": "int8",
}


def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    """
    Function reads pickle file into DataFrame
    and prints size if verbose is True
    """

    df = pd.read_pickle(filename)
    if verbose:
        print("orig_size=" + str(round(
            sys.getsizeof(df) / 1048576, 1)) + " MB")
    df = df.astype(dtypes)  # set types of dataframe
    df = df.rename(columns={"p2a": "date"})  # rename column
    if verbose:
        print("new_size=" + str(round(
            sys.getsizeof(df) / 1048576, 1)) + " MB")
    return df


# Ukol 2: počty nehod v jednotlivých regionech podle druhu silnic
def plot_roadtype(df: pd.DataFrame, fig_location: str = None,
                  show_figure: bool = False):
    """
    Function plots accidents in 4 regions based on road types
    """

    # int road type to string representation
    df["road_types"] = df["p21"].replace(
        {1: "Dvojpruhová komunikácia", 2: "Trojpruhová komunikácia",
         3: "Štvorpruhová komunikácia", 4: "Štvorpruhová komunikácia",
         5: "Viac pruhová komunikácia",
         6: "Rýchlostná komunikácia", 0: "Iný typ komunikácie"})
    data = df.groupby(
        ["region"]).road_types.value_counts().unstack(level=1).reset_index()
    temp_df = data.loc[
        (data['region'] == 'MSK') |
        (data['region'] == 'JHM') |
        (data['region'] == 'ZLK') |
        (data['region'] == 'VYS')]
    melted_df = pd.melt(
        temp_df, ["region"],
        ["Dvojpruhová komunikácia",
         "Trojpruhová komunikácia",
         "Štvorpruhová komunikácia",
         "Viac pruhová komunikácia",
         "Rýchlostná komunikácia",
         "Iný typ komunikácie"])
    sns.set_style("darkgrid")
    graph = sns.catplot(
        data=melted_df, x='region', y='value',
        col="road_types", kind='bar', col_wrap=3,
        sharey=False, palette="viridis", height=3, aspect=1.2)
    graph.fig.subplots_adjust(top=0.85)
    graph.fig.suptitle(
        "Počet nehôd vzhľadom na typ komunikácie v daných krajoch"
        "(MSK, JHM, ZLK, VYS)", fontsize=15)
    graph.set_titles("{col_name}")
    graph.set_axis_labels("Kraj", "Počet nehôd")

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


# Ukol3: zavinění zvěří
def plot_animals(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Function plots accidents in 4 regions based on accident fault
    """
    # int accident fault type to string representation
    df["accident_fault"] = df["p10"].replace(
        {1: "Vodičom", 2: "Vodičom",
         3: "Iné", 4: "Zverou", 5: "Iné",
         6: "Iné", 7: "Iné", 0: "Iné"})
    df["month"] = df["date"].dt.month
    df = df.loc[df['date'].dt.year < 2021]

    data = df.groupby(
        ["region", "month"]).accident_fault.value_counts().unstack(
            level=2).reset_index()

    temp_df = data.loc[
        (data['region'] == 'MSK') |
        (data['region'] == 'JHM') |
        (data['region'] == 'ZLK') |
        (data['region'] == 'VYS')]

    melted_df = pd.melt(
        temp_df, ["region", "month"],
        ["Vodičom", "Zverou", "Iné"])

    sns.set_style("darkgrid")
    graph = sns.catplot(
        data=melted_df, x='month', y='value',
        col="region", hue="accident_fault", kind='bar',
        col_wrap=2, sharey=False, palette="viridis",
        height=4, aspect=1.2, sharex=False)
    graph.fig.subplots_adjust(top=0.85)
    graph.fig.suptitle(
        "Počet nehôd vzhľadom ku ich zavineniam"
        "(MSK, JHM, ZLK, VYS)", fontsize=15)
    graph.set_titles("{col_name}")
    graph.set_axis_labels("mesiac", "Počet nehôd")
    graph._legend.set_title("Zavinenie")

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


# Ukol 4: Povětrnostní podmínky
def plot_conditions(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    """
    Function plots accidents in 4 regions based on weather conditions
    """
    # int accident weather conditions type to string representation
    df["weather"] = df["p18"].replace(
        {1: "Priaznivé", 2: "Hmla",
         3: "Mrholenie", 4: "Dážď", 5: "Sneženie",
         6: "Poľadovica", 7: "Nárazový vietor"})

    df = df[df["p18"] != 0]
    df = df[(df['date'] <= datetime.datetime(2020, 1, 1))]
    temp_df = df.loc[
        (df['region'] == 'MSK') |
        (df['region'] == 'JHM') |
        (df['region'] == 'ZLK') |
        (df['region'] == 'VYS')]
    new_df = temp_df[['date', 'weather', 'region']].copy()

    table = pd.pivot_table(
        new_df, index=['region', 'date'],
        columns=['weather'], aggfunc=len, fill_value=0)
    table = table.reset_index().set_index('date')
    table = table.groupby(['region']).resample("M").sum()
    table = table.stack().reset_index()
    table.rename(columns={0: 'count'}, inplace=True)

    sns.set_style("darkgrid")
    graph = sns.relplot(
        x=table.date, y="count", col="region",
        col_wrap=1, hue=table.weather, data=table,
        palette='gist_ncar', col_order=["MSK", "JHM", "ZLK", "VYS"],
        height=2, aspect=4, kind="line", facet_kws=dict(sharey=False))
    graph.fig.subplots_adjust(top=0.85)
    graph.fig.suptitle(
        "Počet nehôd vzhľadom k cestným podmienkam"
        "(MSK, JHM, ZLK, VYS)", fontsize=15)
    graph.set_titles("{col_name}")
    graph.set_axis_labels("Dátum", "Počet nehôd")
    graph._legend.set_title("Podmienky")

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
