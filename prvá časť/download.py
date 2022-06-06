#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Pattern
import numpy as np
from zipfile import ZipFile
import csv
import gzip
from numpy.core.fromnumeric import shape
from numpy.core.numeric import argwhere, indices
from numpy.lib.function_base import append
import requests
import os
import re
import pickle
from io import TextIOWrapper
from bs4 import BeautifulSoup

# Kromě vestavěných knihoven (os, sys, re, requests …) byste si měli vystačit s: gzip, pickle, csv, zipfile, numpy, matplotlib, BeautifulSoup.
# Další knihovny je možné použít po schválení opravujícím (např ve fóru WIS).


class DataDownloader:
    """ TODO: dokumentacni retezce 

    Attributes:
        headers    Nazvy hlavicek jednotlivych CSV souboru, tyto nazvy nemente!  
        regions     Dictionary s nazvy kraju : nazev csv souboru
    """

    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]

    # types of headers
    header_types = ["U12", "i", "i", "U16", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i",
               "i", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i",
               "i", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i",
               "i", "i", "f", "f", "f", "f", "U16", "U16", "U16", "U16", "U16", "U16", "U16", "U16", "U16", "U16", "U16", "i", "i", "U16", "i", "U3"]

    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    # constructor creates 2 dictionaries for data
    def __init__(self, url="https://ehw.fit.vutbr.cz/izv/", folder="data", cache_filename="data_{}.pkl.gz"):
        self.url = url
        self.folder = folder
        self.cache_filename = cache_filename
        self.mem_dict = {x:np.array([], dtype=self.header_types[i]) for i, x in enumerate(self.headers)}
        self.mem_dict["region"] = [np.array([], dtype="U3")]
        self.return_dict = {x:np.array([], dtype=self.header_types[i]) for i, x in enumerate(self.headers)}
        self.return_dict["region"] = [np.array([], dtype="U3")]

    # method downloads zip files which contains data 
    def download_data(self):
        zip_files = []
        pattern = re.compile(r'data\/(datagis|data-gis)((-?rok|)-?\d{4}|-?\d8-?2021)')
        resp = requests.get('https://ehw.fit.vutbr.cz/izv/')
        resp.headers.get('content-type')
        soup = BeautifulSoup(resp.text, 'html.parser')
        buttons = soup.find_all("button", {"class": "btn btn-sm btn-primary"})

        for btn in buttons:
            if pattern.match(btn['onclick'][10: -2]):
                zip_files.append(btn['onclick'][10: -2])
        for data_url in zip_files:
            data_name = data_url[5:]
            r = requests.get("{}/{}".format(self.url, data_url))
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            with open("{}/{}".format(self.folder, data_name), 'wb') as f:
                f.write(r.content)
    
    # method parses data for specific region and returns dictionary with regions data
    def parse_region_data(self, region):
        read_dictionary = {x:[] for x in self.headers}
        read_dictionary["region"] = []
        region_file_name = self.regions.get(region)
        region_file_name += ".csv"
        
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            self.download_data()
        else:
            self.download_data()
        
        for file in os.scandir(self.folder):
            if file.path.endswith(".zip"):
                with ZipFile(file, 'r') as zf:
                    with zf.open(region_file_name, 'r') as csv_file:
                        reader = csv.reader(TextIOWrapper(csv_file, 'cp1250'), delimiter=';')
                        for row in reader:
                            for i, record in enumerate(row):
                                if (record == "" or record == "XX" or record == "A:" or record == "B:" or record == "D:" or record == "E:"):
                                    read_dictionary[self.headers[i]].append(-1)
                                else:
                                    record = record.replace(",", ".")
                                    read_dictionary[self.headers[i]].append(record)
                            read_dictionary["region"].append(region)

        for key in read_dictionary.keys():
            self.mem_dict[key] = np.append(self.mem_dict[key], read_dictionary[key])
        
        iterator = 0
        for key, val in read_dictionary.items():
            read_dictionary[key] = np.array(val, dtype=self.header_types[iterator])
            iterator += 1

        return read_dictionary
    
    # private method checking whether or not data is stored in memory
    def __memory_cmp_reg(self, reg):
        slice = np.where(self.mem_dict["region"] == reg)
        slice = slice[0]
        
        if slice.size == 0:
            return -1
        else:
            for key in self.return_dict.keys():
                self.return_dict[key] = np.append(self.return_dict[key], self.mem_dict[key][slice[0]:slice[-1]])
            return 0

    # private method checks whether or not data is stored in cache files
    def __cache_cmp_reg(self, reg):

        if self.__memory_cmp_reg(reg) == 0:
            return

        if os.path.exists("{}/{}".format(self.folder, self.cache_filename.format(reg))):
            fp=gzip.open("{}/{}".format(self.folder, self.cache_filename.format(reg)),'rb', compresslevel=1)
            loaded_dict = pickle.load(fp)
            for key in loaded_dict.keys():
                self.return_dict[key] = np.append(self.return_dict[key], loaded_dict[key])
                self.mem_dict[key] = np.append(self.mem_dict[key], loaded_dict[key])
            fp.close()
        else:
            region_dict = self.parse_region_data(reg)
            fp=gzip.open("{}/{}".format(self.folder, self.cache_filename.format(reg)),'wb', compresslevel=1)
            pickle.dump(region_dict ,fp)
            for key in region_dict.keys():
                self.return_dict[key] = np.append(self.return_dict[key], region_dict[key])
                self.mem_dict[key] = np.append(self.mem_dict[key], region_dict[key])
            fp.close()
        
    # method creates return dictionary out of 1 - * regions
    def get_dict(self, regions=None):
        if regions == None or regions == []:
            for region in self.regions.keys():
                self.__cache_cmp_reg(region)
        else:
            for region in regions:
                self.__cache_cmp_reg(region)

        # deletion of duplicated data
        NULL, counts = np.unique(self.return_dict["p1"], return_counts=1)
        indexes_to_delete = np.argwhere(counts>1)

        for key in self.return_dict.keys():
            self.return_dict[key] = np.delete(self.return_dict[key], indexes_to_delete)

        return self.return_dict

# TODO vypsat zakladni informace pri spusteni python3 download.py (ne pri importu modulu)

if (__name__ == '__main__'):
    dictionary = DataDownloader().get_dict(["PHA", "KVK", "ULK"])
    for key, value in dictionary.items():
        print("{}:{}".format(key, len(dictionary[key])))
    print("regions:", np.unique(dictionary["region"]))


