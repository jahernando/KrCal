#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 2018

Plots functions for Kr,

@author: G. Martinez, J.A.hernando
"""

import numpy             as np
import matplotlib.pyplot as plt

import invisible_cities.core.fit_functions as fitf

from invisible_cities.core .core_functions import in_range
from invisible_cities.icaro.hst_functions  import shift_to_bin_centers

from icaro.core.fit_functions import fit_slices_1d_gauss
from icaro.core.fit_functions import expo_seed
from invisible_cities.icaro.hst_functions import labels


default_cmap = 'jet'


def selection_in_band(E, Z, Erange, Zrange, Zfitrange, nsigma = 3.5,
                      Znbins = 50, Enbins =100, plot=True):
    """ returns a selection of the event sin the band of nsigmas around the 
    mean value of the sigma per slice.
    """
    Zfit = Zfitrange

    Zbins = np.linspace(*Zrange, Znbins + 1)
    Ebins = np.linspace(*Erange, Enbins + 1)

    Zcenters = shift_to_bin_centers(Zbins)
    Zerror   = np.diff(Zbins) * 0.5

    sel_e = in_range(E, *Erange)
    mean, sigma, chi2, ok = fit_slices_1d_gauss(Z[sel_e], E[sel_e], Zbins, Ebins, min_entries=5e2)
    ok = ok & in_range(Zcenters, *Zfit)

    def _line_cut(sign):
        x         = Zcenters[ok]
        y         = mean.value[ok] + sign*nsigma * sigma.value[ok]
        yu        = mean.uncertainty[ok]
        seed      = expo_seed(x, y)
        efit  = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
        assert np.all(efit.values != seed)
        return efit.fn

    lowE_cut  = _line_cut(-1.)
    highE_cut = _line_cut(+1.)

    sel_inband = in_range(E, lowE_cut(Z), highE_cut(Z))

    if (plot == False): return sel_inband

    plt.hist2d  (Z, E, (Zbins, Ebins), cmap=default_cmap)
    plt.errorbar(   Zcenters[ok], mean.value[ok],
                sigma.value[ok],     Zerror[ok],
                "kp", label="Kr peak energy $\pm 1 \sigma$")
    f = fitf.fit(fitf.expo, Zcenters[ok], mean.value[ok], (1e4, -1e3))
    plt.plot(Zcenters, f.fn(Zcenters), "r-")
    print(f.values)
    plt.plot    (Zbins,  lowE_cut(Zbins),  "m", lw=2, label="$\pm "+str(nsigma)+" \sigma$ region")
    plt.plot    (Zbins, highE_cut(Zbins),  "m", lw=2)
    plt.legend()
    labels("Drift time (µs)", "S2 energy (pes)", "Energy vs drift")

    return sel_inband
