#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 6 Jul 2018

@author: hernando
"""


import tables            as tb
import numpy             as np

import invisible_cities.core.fit_functions as fitf
import invisible_cities.reco.dst_functions as dstf
import invisible_cities.io  .kdst_io       as kdstio

from table_info import RunInfo
from table_info import MapInfo

from collections import namedtuple

XYMap = namedtuple('XYMap',
                   ('x', 'y', 'values', 'errors'))


def Ecorrection(correction_filename, S2e, X, Y, Z = None, T = None):

    ELT_correction  = dstf.load_lifetime_xy_corrections(correction_filename,
                                                        group = "XYcorrections",
                                                        node  = "Elifetime")

    EGEO_correction = dstf.load_xy_corrections(correction_filename,
                                               group         =  "XYcorrections",
                                               node          =  "Egeometry",
                                               norm_strategy =  "index",
                                               norm_opts     = {"index": (40, 40)})

    # this code is prepared for Time correction along the run
    #Etime_correction = dstf.load_t_corrections(correction_filename,
    #                                          group         = "Corrections",
    #                                          node          = "Etime",
    #                                          norm_strategy = "const",
    #                                          norm_opts     = {"value": 41.5})

    E  = S2e * EGEO_correction(X, Y).value
    if (Z is None): return E

    E = E * ELT_correction(Z, X, Y).value
    if (T is None): return E

    # no time correction jet!
    #t = (T-np.min(T))/60. # in minutes
    #E  = E   * Etime_correction(t).value

    return E

def Qcorrection(correction_filename, S2q, X, Y, Z = None, T = None):

    QLT_correction  = dstf.load_lifetime_xy_corrections(correction_filename,
                                                    group = "XYcorrections",
                                                    node  = "Qlifetime")

    QGEO_correction = dstf.load_xy_corrections(correction_filename,
                                           group         =  "XYcorrections",
                                           node          =  "Qgeometry",
                                           norm_strategy =  "index",
                                           norm_opts     = {"index": (40, 40)})


    # Code is prepared for time corrections along the time
    #Qtime_correction = dstf.load_t_corrections(correction_filename,
    #                                      group         = "Corrections",
    #                                      node          = "Qtime",
    #                                      norm_strategy = "const",
    #                                      norm_opts     = {"value": 41.5})

    Q  = S2q * QGEO_correction(X, Y).value
    if (Z is None): return Q

    Q  = Q   * QLT_correction(Z, X, Y).value
    if (T is None): return Q

    #t = (T-np.min(T))/60. # in minutes
    #Q  = Q   * Qtime_correction(t).value

    return Q


def get_xymap(correction_filename, xymap_name):
    # xymap_name = 'Elifetime', 'Qlifetime', 'Egeometry', 'Qgeometry'
    xymap = dstf.load_dst(correction_filename,
                          group = "XYcorrections",
                          node  = xymap_name)

    x   = np.unique(xymap.x.values)
    y   = np.unique(xymap.y.values)
    values = xymap.factor     .values.reshape(x.size, y.size)
    errors = xymap.uncertainty.values.reshape(x.size, y.size)

    sel = values > 0
    errors[sel]  = errors[sel]*100/values[sel]
    errors[~sel] = 0.

    return XYMap(x, y, values, errors)
