import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as md
from invisible_cities.icaro.mpl_functions import set_plot_labels
from   invisible_cities.core.system_of_units_c import units

from . kr_types import KrLTLimits
from . kr_types import NevtDst
from invisible_cities.icaro. hst_functions import display_matrix
from   invisible_cities.evm  .ic_containers  import Measurement
from invisible_cities.icaro. hst_functions import display_matrix
from icaro.core.fit_functions import conditional_labels

labels = conditional_labels(True)


def figsize(type="small"):
    if type == "S":
        plt.rcParams["figure.figsize"]  = 8, 6
    elif type == "s":
         plt.rcParams["figure.figsize"] = 6, 4
    elif type == "l":
        plt.rcParams["figure.figsize"] = 10, 8
    else:
        plt.rcParams["figure.figsize"] = 12, 10

def plot_xy_density(kdst, krBins, figsize=(14,10)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt_full, *_ = plt.hist2d(kdst.full.X, kdst.full.Y, (krBins.XY, krBins.XY))
    plt.colorbar().set_label("Number of events")
    labels("X (mm)", "Y (mm)", f"full distribution for {krBins.pXY:.1f} mm pitch")

    fig.add_subplot(2, 2, 2)
    nevt_fid, *_ = plt.hist2d(kdst.fid.X, kdst.fid.Y, (krBins.XY, krBins.XY))
    plt.colorbar().set_label("Number of events")
    labels("X (mm)", "Y (mm)", f"fid distribution for {krBins.pXY:.1f} mm pitch")

    fig.add_subplot(2, 2, 3)
    nevt_core, *_ = plt.hist2d(kdst.core.X, kdst.core.Y, (krBins.XY, krBins.XY))
    plt.colorbar().set_label("Number of events")
    labels("X (mm)", "Y (mm)", f"core distribution for {krBins.pXY:.1f} mm pitch")

    fig.add_subplot(2, 2, 4)
    nevt_hcore, *_ = plt.hist2d(kdst.hcore.X, kdst.hcore.Y, (krBins.XY, krBins.XY))
    plt.colorbar().set_label("Number of events")
    labels("X (mm)", "Y (mm)", f"hard core distribution for {krBins.pXY:.1f} mm pitch")
    plt.tight_layout()
    return NevtDst(full  = nevt_full,
                   fid   = nevt_fid,
                   core  = nevt_core,
                   hcore = nevt_hcore)

def plot_s2_vs_z(kdst, krBins, figsize=(14,10)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(kdst.full.Z, kdst.full.E, (krBins.Z, krBins.E))
    plt.colorbar().set_label("Number of events")
    labels("Z (mm)", "E (pes)", f" full ")

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(kdst.fid.Z, kdst.fid.E, (krBins.Z, krBins.E))
    plt.colorbar().set_label("Number of events")
    labels("Z (mm)", "E (pes)", f" fid ")

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(kdst.core.Z, kdst.core.E, (krBins.Z, krBins.E))
    plt.colorbar().set_label("Number of events")
    labels("Z (mm)", "E (pes)", f" core ")

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(kdst.hcore.Z, kdst.hcore.E, (krBins.Z, krBins.E))
    plt.colorbar().set_label("Number of events")
    labels("Z (mm)", "E (pes)", f" hard core Z")
    plt.tight_layout()


def plot_s1_vs_z(kdst, krBins, figsize=(14,10)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(kdst.full.Z, kdst.full.S1, (krBins.Z, krBins.S1))
    plt.colorbar().set_label("Number of events")
    labels("Z (mm)", "S1 (pes)", f"full S1 vs Z")

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(kdst.fid.Z, kdst.fid.S1, (krBins.Z, krBins.S1))
    plt.colorbar().set_label("Number of events")
    labels("Z (mm)", "S1 (pes)", f"fid S1 vs Z")

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(kdst.core.Z, kdst.core.S1, (krBins.Z, krBins.S1))
    plt.colorbar().set_label("core Number of events")
    labels("Z (mm)", "S1 (pes)", f"S1 vs Z")

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(kdst.hcore.Z, kdst.hcore.S1, (krBins.Z, krBins.S1))
    plt.colorbar().set_label("hard core Number of events")
    labels("Z (mm)", "S1 (pes)", f"S1 vs Z")
    plt.tight_layout()

def plot_s2_vs_s1(kdst, krBins, figsize=(14,10)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(kdst.full.S1, kdst.full.E, (krBins.S1, krBins.E))
    plt.colorbar().set_label("Number of events")
    labels("S1 (pes)", "S2 (pes)", f"full S2 vs S1")

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(kdst.fid.S1, kdst.fid.E, (krBins.S1, krBins.E))
    plt.colorbar().set_label("Number of events")
    labels("S1 (pes)", "S2 (pes)", f"fid S2 vs S1")
    fig.add_subplot(2, 2, 3)

    nevt, *_  = plt.hist2d(kdst.core.S1, kdst.core.E, (krBins.S1, krBins.E))
    plt.colorbar().set_label("core Number of events")
    labels("S1 (pes)", "S2 (pes)", f"core S2 vs S1")
    fig.add_subplot(2, 2, 4)

    nevt, *_  = plt.hist2d(kdst.hcore.S1, kdst.hcore.E, (krBins.S1, krBins.E))
    plt.colorbar().set_label("hard core Number of events")
    labels("S1 (pes)", "S2 (pes)", f"hard core S2 vs S1")
    plt.tight_layout()


def plot_lifetime_T(kfs, timeStamps, ltlim=(2000, 3000),  figsize=(12,6)):
    ez0s = [kf.par[0] for kf in kfs]
    lts = [np.abs(kf.par[1]) for kf in kfs]
    u_ez0s = [kf.err[0] for kf in kfs]
    u_lts = [kf.err[1] for kf in kfs]
    plt.figure(figsize=figsize)
    ax=plt.gca()
    xfmt = md.DateFormatter('%d-%m %H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    plt.errorbar(timeStamps, lts, u_lts, fmt="kp", ms=7, lw=3)
    plt.xlabel('date')
    plt.ylabel('Lifetime (mus)')
    plt.ylim(ltlim)
    plt.xticks( rotation=25 )

def display_lifetime_maps(Escale : Measurement,
                          ELT: Measurement,
                          kltl : KrLTLimits,
                          XYcenters : np.array,
                          cmap = "jet",
                          mask = None):

    """Display lifetime maps: the mask allow to specify channels
    to be masked out (usually bad channels)
    """

    #fig = plt.figure(figsize=figsize)
    #fig.add_subplot(2, 2, 1)
    plt.subplot(2, 2, 1)
    *_, cb = display_matrix(XYcenters, XYcenters, Escale.value, mask,
                            vmin = kltl.Es.min,
                            vmax = kltl.Es.max,
                            cmap = cmap,
                            new_figure = False)
    cb.set_label("Energy scale at z=0 (pes)")
    labels("X (mm)", "Y (mm)", "Energy scale")

    #fig.add_subplot(2, 2, 2)
    plt.subplot(2, 2, 2)
    *_, cb = display_matrix(XYcenters, XYcenters, Escale.uncertainty, mask,
                        vmin = kltl.Eu.min,
                        vmax = kltl.Eu.max,
                        cmap = cmap,
                        new_figure = False)
    cb.set_label("Relative energy scale uncertainty (%)")
    labels("X (mm)", "Y (mm)", "Relative energy scale uncertainty")

    #fig.add_subplot(2, 2, 3)
    plt.subplot(2, 2, 3)
    *_, cb = display_matrix(XYcenters, XYcenters, ELT.value, mask,
                        vmin = kltl.LT.min,
                        vmax = kltl.LT.max,
                        cmap = cmap,
                        new_figure = False)
    cb.set_label("Lifetime (µs)")
    labels("X (mm)", "Y (mm)", "Lifetime")

    #fig.add_subplot(2, 2, 4)
    plt.subplot(2, 2, 4)
    *_, cb = display_matrix(XYcenters, XYcenters, ELT.uncertainty, mask,
                        vmin = kltl.LTu.min,
                        vmax = kltl.LTu.max,
                        cmap = cmap,
                        new_figure = False)
    cb.set_label("Relative lifetime uncertainty (%)")
    labels("X (mm)", "Y (mm)", "Relative lifetime uncertainty")

    plt.tight_layout()


def double_hist(h1, h2, binning, label0="Original", label1="Filtered", **kwargs):
    plt.hist(h1, binning, label=label0, alpha=0.5, color="g", **kwargs)
    plt.hist(h2, binning, label=label1, alpha=0.5, color="m", **kwargs)
    plt.legend()

def dst_compare_vars(dst1, dst2):
    dst    = dst1
    subdst = dst2

    plt.figure(figsize=(20, 15))

    plt.subplot(3, 4, 1)
    double_hist(dst.nS2, subdst.nS2, np.linspace(0, 5, 6))
    plt.yscale("log")
    labels("Number of S2s", "Entries", "# S2")

    plt.subplot(3, 4, 2)
    double_hist(dst.S1e, subdst.S1e, np.linspace(0, 50, 51))
    labels("S1 integral (pes)", "Entries", "S1 energy")

    plt.subplot(3, 4, 3)
    double_hist(dst.S1w, subdst.S1w, np.linspace(0, 600, 25))
    labels("S1 width (ns)", "Entries", "S1 width")

    plt.subplot(3, 4, 4)
    double_hist(dst.S1h, subdst.S1h, np.linspace(0, 15, 31))
    labels("S1 height (pes)", "Entries", "S1 height")

    plt.subplot(3, 4, 5)
    double_hist(dst.Nsipm, subdst.Nsipm, np.linspace(0, 100, 51))
    labels("Number of SiPMs", "Entries", "# SiPMs")

    plt.subplot(3, 4, 6)
    double_hist(dst.S2e, subdst.S2e, np.linspace(0, 25e3, 101))
    labels("S2 integral (pes)", "Entries", "S2 energy")

    plt.subplot(3, 4, 7)
    double_hist(dst.S2w, subdst.S2w, np.linspace(0, 50, 26))
    labels("S2 width (µs)", "Entries", "S2 width")

    plt.subplot(3, 4, 8)
    double_hist(dst.S2h, subdst.S2h, np.linspace(0, 1e4, 101))
    labels("S2 height (pes)", "Entries", "S2 height")

    plt.subplot(3, 4, 9)
    double_hist(dst.Z, subdst.Z, np.linspace(0, 600, 101))
    labels("Drift time (µs)", "Entries", "Drift time")

    plt.subplot(3, 4, 10)
    double_hist(dst.X, subdst.X, np.linspace(-200, 200, 101))
    labels("X (mm)", "Entries", "X")

    plt.subplot(3, 4, 11)
    double_hist(dst.Y, subdst.Y, np.linspace(-200, 200, 101))
    labels("Y (mm)", "Entries", "Y")

    plt.subplot(3, 4, 12)
    double_hist(dst.S2q, subdst.S2q, np.linspace(0, 5e3, 101))
    labels("Q (pes)", "Entries", "S2 charge")

    plt.tight_layout()
