import numpy as np
import tables as tb
import pandas as pd
import glob

from   invisible_cities.io.dst_io  import load_dsts
from   invisible_cities.core.core_functions import loc_elem_1d
from   invisible_cities.icaro. hst_functions import shift_to_bin_centers
import datetime

from   . kr_types import KrEvent
from   . kr_types import DstEvent
from   . kr_types import ExyzBins, ExyzNBins, ExyzRanges
from   . kr_types import KrTimes, KrBins, KrNBins, KrRanges

from tables import NoSuchNodeError
from tables import HDF5ExtError
import warnings



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
    except IOError:
        print(f' warning:  {filename} does not exist')


def load_dsts(dst_list, group, node):
    dsts = [load_dst(filename, group, node) for filename in dst_list]
    return pd.concat(dsts)


def file_numbers_from_file_range(file_range):
    numbers = range(*file_range)
    N=[]
    for number in numbers:
        if number < 10:
            N.append(f"000{number}")
        elif 10 <= number < 100:
            N.append(f"00{number}")
        elif 100 <= number < 1000:
            N.append(f"0{number}")
        else:
            N.append(f"{number}")

    return N


def kr_event(dst):
    dst_time = dst.sort_values('event')
    return KrEvent(X  = dst.X.values,
                   Y  = dst.Y.values,
                   Z  = dst.Z.values,
                   T  = dst_time.time.values,
                   E  = dst.S2e.values,
                   S1 = dst.S1e.values,
                   Q  = dst.S2q.values)


def kr_bins(Zrange  = ( 100,  550),
            XYrange = (-220,  220),
            Erange  = ( 2e3, 15e3),
            S1range = (   0,   50),
            Qrange  = ( 100, 1500),
            Znbins        =   10,
            XYnbins       =   30,
            Enbins        =   50,
            S1nbins       =   10,
            Qnbins        =   25):

    Zbins      = np.linspace(* Zrange,  Znbins + 1)
    Ebins      = np.linspace(* Erange,  Enbins + 1)
    S1bins     = np.linspace(* S1range,  S1nbins + 1)
    Qbins      = np.linspace(* Qrange,  Qnbins + 1)
    XYbins     = np.linspace(*XYrange, XYnbins + 1)
    XYcenters  = shift_to_bin_centers(XYbins)
    XYpitch    = np.diff(XYbins)[0]

    exyzBins   = ExyzBins(E = Ebins, S1=S1bins, Q = Qbins, Z = Zbins,
                          XY = XYbins, cXY = XYcenters, pXY = XYpitch)

    return exyzBins


def kr_ranges_and_bins(Zrange  = ( 100,  550),
                       XYrange = (-220,  220),
                       Erange  = ( 2e3, 15e3),
                       S1range = (   0,   50),
                       Qrange  = ( 100, 1500),
                       Znbins        =   10,
                       XYnbins       =   30,
                       Enbins        =   50,
                       S1nbins       =   10,
                       Qnbins        =   25):

    exyzBins = kr_bins(Zrange, XYrange, Erange,
                       S1range, Qrange, Znbins,
                       XYnbins, Enbins, S1nbins,
                       Qnbins)
    exyzNBins  = ExyzNBins( E = Enbins, S1=S1nbins, Q = Qnbins, Z = Znbins,
                            XY = XYnbins)
    exyzRanges = ExyzRanges(E = Erange, S1=S1range, Q = Qrange, Z = Zrange,
                            XY = XYrange)


    return exyzRanges, exyzNBins, exyzBins


def kr_times_ranges_and_bins(dst,
                        Zrange  = ( 100,  550),
                        XYrange = (-220,  220),
                        Erange  = ( 2e3, 15e3),
                        S1range = (   0,   50),
                        Qrange  = ( 100, 1500),
                        Znbins        =   10,
                        XYnbins       =   30,
                        Enbins        =   50,
                        S1nbins       =   10,
                        Qnbins        =   25,
                        nStimeprofile = 3600
                       ):

    exyzRanges, exyzNBins, exyzBins = kr_ranges_and_bins(Zrange, XYrange, Erange,
                                                         S1range, Qrange, Znbins,
                                                         XYnbins, Enbins, S1nbins,
                                                         Qnbins)

    dst_time = dst.sort_values('event')
    T       = dst_time.time.values
    tstart  = T[0]
    tfinal  = T[-1]
    Trange  = (tstart,tfinal)

    ntimebins  = int( np.floor( ( tfinal - tstart) / nStimeprofile) )
    Tnbins     = np.max([ntimebins, 1])
    Tbins      = np.linspace( tstart, tfinal, ntimebins+1)

    krNBins  = KrNBins(E = exyzNBins.E, S1=exyzNBins.S1,
                           Q = exyzNBins.Q, Z = exyzNBins.Z,
                           XY = exyzNBins.XY, T = Tnbins)
    krRanges = KrRanges(E = exyzRanges.E, S1=exyzRanges.S1,
                            Q = exyzRanges.Q, Z = exyzRanges.Z,
                            XY= exyzRanges.XY, T = Trange)
    krBins   = KrBins(E = exyzBins.E, S1=exyzBins.S1,
                          Q = exyzBins.Q, Z = exyzBins.Z,
                          XY = exyzBins.XY, cXY = exyzBins.cXY, pXY = exyzBins.pXY,
                          T = Tbins)

    times      = [np.mean([Tbins[t],Tbins[t+1]]) for t in range(Tnbins)]
    TL         = [(Tbins[t],Tbins[t+1]) for t in range(Tnbins)]
    timeStamps = list(map(datetime.datetime.fromtimestamp, times))
    krTimes    = KrTimes(times = times, timeStamps = timeStamps, TL = TL)

    return krTimes, krRanges, krNBins, krBins


def fiducial_volumes(dst,
                     R_full   = 200,
                     R_fid    = 150,
                     R_core   = 100,
                     R_hcore  =  50):


    dst_full   = dst[dst.R < R_full]
    dst_fid    = dst[dst.R < R_fid]
    dst_core   = dst[dst.R < R_core]
    dst_hcore  = dst[dst.R < R_hcore]

    n_dst      = len(dst)
    n_full     = len(dst_full)
    n_fid      = len(dst_fid)
    n_core     = len(dst_core)
    n_hcore    = len(dst_hcore)

    eff_full   = n_full  / n_dst
    eff_fid    = n_fid   / n_dst
    eff_core   = n_core  / n_dst
    eff_hcore  = n_hcore / n_dst

    print(f" nfull : {n_full}: eff_full = {eff_full} ")
    print(f" nfid : {n_fid}: eff_fid = {eff_fid} ")
    print(f" ncore : {n_core}: eff_core = {eff_core} ")
    print(f" nhcore : {n_hcore}: eff_hcore = {eff_hcore} ")

    kdst= DstEvent(full  = kr_event(dst_full),
                   fid   = kr_event(dst_fid),
                   core  = kr_event(dst_core),
                   hcore = kr_event(dst_hcore))
    return kdst

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def read_dsts(path_to_dsts):
    filenames = glob.glob(path_to_dsts+'/*')
    dstdf = load_dsts(filenames, group='DST', node='Events')
    nsdf  = load_dsts(filenames, group='Extra', node='nS12')
    return nsdf, dstdf


def bin_ratio(array, bins, xbin):
    return array[loc_elem_1d(bins, xbin)] / np.sum(array)


def bin_to_last_ratio(array, bins, xbin):
    return np.sum(array[loc_elem_1d(bins, xbin): -1]) / np.sum(array)


def divide_np_arrays(num, denom):
    assert len(num) == len(denom)
    ok    = denom > 0
    ratio = np.zeros(len(denom))
    np.divide(num, denom, out=ratio, where=ok)
    return ratio
