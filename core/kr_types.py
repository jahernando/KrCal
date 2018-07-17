import numpy as np

from typing      import Tuple
from typing      import Dict
from typing      import List

from dataclasses import dataclass

from   invisible_cities.types.ic_types import minmax
from   invisible_cities.evm  .ic_containers  import Measurement


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
    """Geometry fields of a Krypton Event"""
    X  : np.array
    Y  : np.array
    Z  : np.array

@dataclass
class ExyzEvent(XyzEvent):
    """Geometry + Energy -- Krypton Event"""
    E  : np.array
    S1 : np.array
    Q  : np.array


@dataclass
class KrEvent(ExyzEvent):
    """Geometry + Energy + time--  Krypton Event"""
    T  : np.array


@dataclass
class XyzBins:
    Z   : np.array
    XY  : np.array
    cXY : np.array  # bin centers
    pXY : float     # pitch


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


@dataclass
class  KrTimes:
    times      : List[float]
    timeStamps : List[float]
    TL         : List[Tuple[float]]


@dataclass
class ExyAvg:
    E  : np.array
    Eu : np.array


@dataclass
class DstEvent:
    full  : KrEvent
    fid   : KrEvent
    core  : KrEvent
    hcore : KrEvent


@dataclass
class NevtDst:
    full  : np.array
    fid   : np.array
    core  : np.array
    hcore : np.array


@dataclass
class Ranges:
    lower  : Tuple[float]
    upper  : Tuple[float]


@dataclass
class XYRanges:
    X  : Tuple[float]
    Y  : Tuple[float]


@dataclass
class KrFit:
    par  : np.array
    err  : np.array
    chi2 : float


@dataclass
class KrLTSlices:
    Es    : np.array
    LT    : np.array
    chi2  : np.array
    valid : np.array


@dataclass
class KrLTLimits:
    Es  : minmax
    LT  : minmax
    Eu  : minmax
    LTu : minmax


@dataclass
class KrMeanAndStd:
    mu    : float
    std   : float
    mu_u  : float
    std_u : float


@dataclass
class KrMeanStdMinMax(KrMeanAndStd):
    min     : float
    max     : float
    min_u    : float
    max_u    : float


@dataclass
class KrMeansAndStds:
    mu    : np.array
    std   : np.array
    mu_u  : np.array
    std_u : np.array
