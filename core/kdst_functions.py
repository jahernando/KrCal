#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 2018

Plots functions for Kr,

@author: G. Martinez, J.J. Gomez-Cadenas, J.A.hernando
"""

import numpy             as np
import scipy.stats       as stats
import matplotlib.pyplot as plt
import tables            as tb
import myhst_functions   as hst
import pandas            as pd

import invisible_cities.core.fit_functions as fitf
import invisible_cities.reco.dst_functions as dstf

from invisible_cities.core .core_functions import in_range
from invisible_cities.icaro.hst_functions  import shift_to_bin_centers

from icaro.core.fit_functions import fit_slices_1d_gauss
from icaro.core.fit_functions import expo_seed
from icaro.core.fit_functions import conditional_labels
#labels = conditional_labels(with_titles)
from invisible_cities.icaro.hst_functions import labels

import tables as tb
from tables import NoSuchNodeError
from tables import HDF5ExtError
import warnings


#---------- load dsts

def load_dst(filename, group, node):
    try:
        with tb.open_file(filename) as h5in:
            try:
                table = getattr(getattr(h5in.root, group), node).read()
                return pd.DataFrame.from_records(table)
            except NoSuchNodeError:
                print(f' warning:  {filename} not of kdst type')
    except HDF5ExtError:
        print(f' warning:  {filename} corrupted')

def load_dsts(dst_list, group, node):
    dsts = [load_dst(filename, group, node) for filename in dst_list]
    return pd.concat(dsts)

#--- other utilities

def kdst_unique_events(dst):
    unique_events = ~dst.event.duplicated()

    number_of_S2s_full  = np.size         (unique_events)
    number_of_evts_full = np.count_nonzero(unique_events)

    #print(f"Total number of S2s   : {number_of_S2s_full} ")
    #print(f"Total number of events: {number_of_evts_full}")

    return number_of_S2s_full, number_of_evts_full


def kdst_write(dst, filename):
    # Unfortunately, this method can't set a specific name to the table or its title.
    # It also includes an extra column ("index") which I can't manage to remove.
    dst.to_hdf(filename,
              key     = "DST"  , mode         = "w",
              format  = "table", data_columns = True,
              complib = "zlib" , complevel    = 4)

    # Workaround to re-establish the name of the table and its title
    with tb.open_file(filename, "r+") as f:
        f.rename_node(f.root.DST.table, "Events")
        f.root.DST.Events.title = "Events"


# selections (this should be in a different module -selections-)

def selection_info(sel, comment=''):
    nsel   = np.sum(sel)
    effsel = 100.*nsel/(1.*len(sel))
    print(f"Total number of selected candidates {comment}: {nsel} ({effsel:.1f} %)" )
