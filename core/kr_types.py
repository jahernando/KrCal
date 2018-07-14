import numpy as np

from typing      import NamedTuple
from typing      import Tuple
from typing      import Dict
from typing      import List

from   invisible_cities.evm  .ic_containers  import Measurement
from   invisible_cities.types.ic_types import minmax
from invisible_cities.icaro. hst_functions import shift_to_bin_centers
import datetime
from dataclasses import dataclass

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


@dataclass
class S1D:
    """S1 description"""
    E  : Measurement
    W  : Measurement
    H  : Measurement
    R  : Measurement # R = H/E
    T  : Measurement


@dataclass
class S2D:
    """S2 description"""
    E  : Measurement
    W  : Measurement
    Q  : Measurement
    N  : Measurement # NSipm
    X  : Measurement
    Y  : Measurement

@dataclass
class XyzEvent:
    """Collects the crucial fields of a Krypton Event"""
    X  : np.array
    Y  : np.array
    Z  : np.array

@dataclass
class ExyzEvent(XyzEvent):
    """Collects the crucial fields of a Krypton Event"""
    E  : np.array
    S1 : np.array
    Q  : np.array


@dataclass
class KrEvent(ExyzEvent):
    """Collects the crucial fields of a Krypton Event"""
    T  : np.array


@dataclass
class XyzBins:
    Z   : np.array
    XY  : np.array
    cXY : np.array
    pXY : float


@dataclass
class ExyzBins(XyzBins):
    E   : np.array
    S1  : np.array
    Q   : np.array


@dataclass
class KrBins(ExyzBins):
    T   : np.array


@dataclass
class XyzNBins:
    Z  : int
    XY : int

@dataclass
class ExyzNBins(XyzNBins):
    E  : int
    S1 : int
    Q  : int


@dataclass
class KrNBins(ExyzNBins):
    T  : int


@dataclass
class XyzRanges:
    Z  : Tuple[float]
    XY : Tuple[float]


@dataclass
class ExyzRanges(XyzRanges):
    E  : Tuple[float]
    S1 : Tuple[float]
    Q  : Tuple[float]


@dataclass
class KrRanges(ExyzRanges):
    T  : Tuple[float]


class  KrTimes(NamedTuple):
    times      : List[float]
    timeStamps : List[float]
    TL         : List[Tuple[float]]


class DstEvent(NamedTuple):
    full  : KrEvent
    fid   : KrEvent
    core  : KrEvent
    hcore : KrEvent


class NevtDst(NamedTuple):
    full  : np.array
    fid   : np.array
    core  : np.array
    hcore : np.array


class Ranges(NamedTuple):
    lower  : Tuple[float]
    upper  : Tuple[float]


class XYRanges(NamedTuple):
    X  : Tuple[float]
    Y  : Tuple[float]


class KrFit(NamedTuple):
    par  : np.array
    err  : np.array
    chi2 : float


class KrLTSlices(NamedTuple):
    Es    : np.array
    LT    : np.array
    chi2  : np.array
    valid : np.array

class KrLTLimits(NamedTuple):
    Es  : minmax
    LT  : minmax
    Eu  : minmax
    LTu : minmax


class KrMeanAndStd(NamedTuple):
    mu    : float
    std   : float
    mu_u  : float
    std_u : float


class KrMeansAndStds(NamedTuple):
    mu    : np.array
    std   : np.array
    mu_u  : np.array
    std_u : np.array
