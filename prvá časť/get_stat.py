#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
from matplotlib import colors
from numpy.core.numeric import indices
from numpy.lib.arraysetops import unique
# povolene jsou pouze zakladni knihovny (os, sys) a knihovny numpy, matplotlib a argparse

from download import DataDownloader

# parsing of arguments
parser = argparse.ArgumentParser(description='--fig_location : sets path of output graph\n --show_figure : shows figure while the script is running, no param argument')
parser.add_argument("--fig_location")
parser.add_argument("--show_figure", action='store_true')
args = parser.parse_args()


def plot_stat(data_source,
              fig_location=None,
              show_figure=False):
    # getting all regions and critical data
    regions = data_source["region"]
    p24 = data_source["p24"]

    # splitting of arrays and intialization of used variables
    NULL, indices = (np.unique(regions, return_index=1))
    indices = sorted(indices)
    regy = np.split(regions, indices[1:])
    nehody = np.split(p24, indices[1:])
    regy = [x[0] for x in regy]
    nehody = [x for x in nehody]
    nehody2 = [x for x in nehody]
    count_region = {"{}".format(x):0 for x in range(6)}
    count_regions = {"{}".format(x):0 for x in range(6)}

    # filling count of occurances to dictionary for one region at the time and also for every region
    for index, nehoda in enumerate(nehody):
        unique, counts = np.unique(nehody[index], return_counts=1)
        if 0 not in unique:
            count_region["0"] = 0
            count_regions["0"] += 0
        if 1 not in unique:
            count_region["1"] = 0
            count_regions["1"] += 0
        if 2 not in unique:
            count_region["2"] = 0
            count_regions["2"] += 0
        if 3 not in unique:
            count_region["3"] = 0
            count_regions["3"] += 0
        if 4 not in unique:
            count_region["4"] = 0
            count_regions["4"] += 0
        if 5 not in unique:
            count_region["5"] = 0
            count_regions["5"] += 0
        
        for i, x in enumerate(unique):
            count_region["{}".format(unique[i])] = counts[i]
            count_regions["{}".format(unique[i])] += counts[i]
    
        nehody[index] = [x for x in count_region.values()]
        nehody[index].append(nehody[index].pop(0))
    
    region_counts = count_regions.values()
    region_counts = list(region_counts)
    region_counts.append(region_counts.pop(0))
    # setting 0.0 values to np.nan
    for idx, nehoda in enumerate(nehody):
        nehody2[idx] = [x/region_counts[i]*100 for i, x in enumerate(nehoda)]
        nehody2[idx] = np.array(nehody2[idx])
        nehody2[idx][nehody2[idx] == 0] = np.nan

    # plotting of graphs
    plt.figure(figsize=(12,6))
    plt.subplot(2, 1, 1)
    plt.colorbar(mpl.cm.ScalarMappable(norm=colors.LogNorm(vmin=10**0, vmax=10**5)), label="Počet nehôd")
    plt.tight_layout(pad=3)
    plt.title("Absolútne hodnoty")
    my_xticks = regy
    my_yticks = ["Prerušovaná žltá", "Semafor mimo prevádzku", "Dopravnými značkami", "Prenosnými d. značkami", "Nevyznačená", "Žiadna úprava"]
    plt.xticks(range(len(my_xticks)), my_xticks)
    plt.yticks(range(6), my_yticks)
    plt.imshow(np.transpose(np.array(nehody)), norm=colors.LogNorm(vmin=10**0, vmax=10**5))
    plt.subplot(2, 1, 2)
    plt.colorbar(mpl.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=100), cmap='plasma'), label="Podiel nehôd pre danú príčinu [%]")
    my_yticks = ["Prerušovaná žltá", "Semafor mimo prevádzku", "Dopravnými značkami", "Prenosnými d. značkami", "Nevyznačená", "Žiadna úprava"]

    plt.title("Hodnoty relatívne voči príčine")
    plt.xticks(range(len(my_xticks)), my_xticks)
    plt.yticks(range(6), my_yticks)
    plt.imshow(np.transpose(np.array(nehody2)), norm=colors.Normalize(vmin=0, vmax=100), cmap='plasma')
    
    # recursive creating of file
    if fig_location is not None:
        index = fig_location[::-1].find('/')
        if index == -1:
            file_name = fig_location
        else:
            file_name = fig_location[-index:]
            os.makedirs(fig_location[:-index], exist_ok=True)
        plt.savefig(fig_location)
    
    if show_figure == True:
        plt.show()
    
# TODO pri spusteni zpracovat argumenty

if (__name__ == '__main__'):

    data = DataDownloader().get_dict()
    if args.show_figure == True:
        plot_stat(data, fig_location=args.fig_location, show_figure=True)
    else:
        plot_stat(data, fig_location=args.fig_location)