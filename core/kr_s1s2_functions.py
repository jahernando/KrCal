import numpy as np

import matplotlib.pyplot as plt
from   invisible_cities.core.core_functions import weighted_mean_and_std
from   invisible_cities.core.core_functions import loc_elem_1d
from   invisible_cities.core.system_of_units_c import units

from . kr_core_functions import bin_ratio
from . kr_core_functions import bin_to_last_ratio
from . kr_core_functions import divide_np_arrays
from   invisible_cities.evm  .ic_containers  import Measurement

from . kr_types import S1D, S2D

dst =['event', 'time', 's1_peak', 's2_peak', 'nS1', 'nS2',
      'S1w', 'S1h', 'S1e', 'S1t', 'S2w', 'S2h', 'S2e', 'S2q', 'S2t',
      'Nsipm', 'DT', 'Z',
      'Zrms', 'X', 'Y', 'R', 'Phi', 'Xrms', 'Yrms']

def ns12(nsdf, type='S1'):
    var = nsdf.nS1
    if type == 'S2' :
        var = nsdf.nS2

    hns1, bins  = histo_ns12(var)
    print(' 0S2/tot  = {} 1S2/tot = {} 2S2/tot = {}'.format(bin_ratio(hns1, bins, 0),
           bin_ratio(hns1, bins, 1),
           bin_ratio(hns1, bins, 2)))


def ns1_stats(nsdf):
    mu, std = weighted_mean_and_std(nsdf.nS1, np.ones(len(nsdf.nS1)))
    hist, bins = np.histogram(nsdf.nS1, bins = 5, range=(0,5))
    s1r = [bin_ratio(hist, bins, i) for i in range(0,4)]
    s1r.append(bin_to_last_ratio(hist, bins, 4))
    return mu, std, s1r


def ns2_stats(nsdf):
    mu, std = weighted_mean_and_std(nsdf.nS2, np.ones(len(nsdf.nS2)))
    hist, bins = np.histogram(nsdf.nS2, bins = 5, range=(0,5))
    s1r = [bin_ratio(hist, bins, i) for i in range(0,3)]
    s1r.append(bin_to_last_ratio(hist, bins, 3))
    return mu, std, s1r


def print_ns12_stats(mu, std, s1r):
    print('ns12: mean = {:5.2f} std = {:5.2f}'.format(mu, std))
    print('ns12 : fraction')
    print('\n'.join('{}: {:5.2f}'.format(*k) for k in enumerate(s1r)))


def ns1s(rnd, figsize=(6,6)):
    vals = rnd.values()
    labels = [x.label for x in vals]
    dsts   = [x.dst for x in vals]
    fig = plt.figure(figsize=figsize) # Creates a new figure

    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('n S1',fontsize = 11)
    ax.set_ylabel('Frequency', fontsize = 11)
    hns1, bins, _ = ax.hist([df.ns1.values for df in dsts], bins = 20, range=(0,20),
            histtype='step',
            label=labels,
            linewidth=1.5)
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)


def get_stats(rnd, i, f, r=False):
    xr = list(range(i,f))
    if r:
        xr = list(reversed(xr))
    stats = [ns1_stats(rnd[i].ns) for i in xr]
    return stats, xr


def plot_srs(rnd, ri, rf, reverse=False):
    stats, xr =get_stats(rnd, ri, rf, reverse)
    srs =[stats[i][2] for i in range(len(stats))]

    for j, i in enumerate(xr):
        plt.plot(srs[j], label=rnd[i].label)
    plt.xlabel('ns1')
    plt.ylabel('fraction')
    plt.grid(True)
    plt.legend(fontsize= 10, loc='upper right')


def plot_mus(rnd, ri, rf, reverse=False):
    stats, xr =get_stats(rnd, ri, rf, reverse)
    mus = [stats[i][0] for i in range(len(stats))]

    plt.plot(xr, mus)
    plt.xlabel('run number')
    plt.ylabel('mean')
    plt.grid(True)
        #plt.legend(fontsize= 10, loc='upper right')


def print_stats(rn, rnd, mu, var, s1r):
    lbl = rnd[rn].label
    print(lbl)
    print('ns1: mean = {:5.2f} sigma = {:5.2f}'.format(mu, np.sqrt(var)))
    print('ns1 : fraction')
    print('\n'.join('{}: {:5.2f}'.format(*k) for k in enumerate(s1r)))


def histo_ns12(ns1,
               xlabel='n S12', ylabel='arbitrary units',
               title = 'ns12', norm=1, fontsize = 11, figsize=(6,6)):
    fig = plt.figure(figsize=figsize) # Creates a new figure

    ax = fig.add_subplot(1, 1, 1)
    #mu, var = weighted_mean_and_var(ns1, np.ones(len(ns1)))
    ax.set_xlabel(xlabel,fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)
    ax.set_title(title, fontsize = 12)
    hns1, bins, _ =ax.hist(ns1,
            normed=norm,
            bins = 10,
            range=(0,10),
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label='# S1 candidates')
    #ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)
    return hns1, bins


def s1_means_and_vars(dst):

    hr = divide_np_arrays(dst.S1h.values, dst.S1e.values)
    return S1D(E = Measurement(*weighted_mean_and_std(dst.S1e.values,
                                                      np.ones(len(dst)))),
               W = Measurement(*weighted_mean_and_std(dst.S1w,
                                                      np.ones(len(dst)))),
               H = Measurement(*weighted_mean_and_std(dst.S1h,
                                                      np.ones(len(dst)))),
               R = Measurement(*weighted_mean_and_std(hr,
                                                      np.ones(len(dst)))),
               T = Measurement(*weighted_mean_and_std(dst.S1t,
                                                      np.ones(len(dst)))))

def s2_means_and_vars(dst):


    return S2D(E = Measurement(*weighted_mean_and_std(dst.S2e.values,
                                                      np.ones(len(dst)))),
               W = Measurement(*weighted_mean_and_std(dst.S2w,
                                                      np.ones(len(dst)))),
               Q = Measurement(*weighted_mean_and_std(dst.S2q,
                                                   np.ones(len(dst)))),
               N = Measurement(*weighted_mean_and_std(dst.Nsipm,
                                                np.ones(len(dst)))),
               X = Measurement(*weighted_mean_and_std(dst.X,
                                             np.ones(len(dst)))),
               Y = Measurement(*weighted_mean_and_std(dst.Y,
                                              np.ones(len(dst)))))



def plot_s1histos(dst, s1d, bins=20, figsize=(12,12)):

    fig = plt.figure(figsize=figsize) # Creates a new figure
    ax = fig.add_subplot(3, 2, 1)

    ax.set_xlabel('S1 energy (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)#ylabel
    ax.hist(dst.S1e,
            range=(0,40),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.E.value, s1d.E.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 2)
    ax.set_xlabel(r'S1 width ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(dst.S1w,
            range=(0,500),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.W.value, s1d.W.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 3)
    ax.set_xlabel(r'S1 height (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(dst.S1h,
            range=(0,10),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.H.value, s1d.H.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 4)
    hr = divide_np_arrays(dst.S1h.values, dst.S1e.values)

    ax.set_xlabel(r'height / energy ',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(hr,
            range=(0,1),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.R.value, s1d.R.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 5)
    ax.set_xlabel(r'S1 time ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(dst.S1t / units.mus,
            range=(0,600),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.R.value, s1d.R.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 6)
    plt.hist2d(dst.S1t/units.mus, dst.S1e, bins=10, range=((0,600),(0,30)))
    plt.colorbar()
    ax.set_xlabel(r'S1 time ($\mu$s) ',fontsize = 11) #xlabel
    ax.set_ylabel('S1 height (pes)', fontsize = 11)
    plt.grid(True)

    plt.tight_layout()


def plot_s2histos(df, s2d, bins=20, emin=3000, emax=15000, figsize=(12,12)):

    fig = plt.figure(figsize=figsize) # Creates a new figure
    ax = fig.add_subplot(3, 2, 1)

    ax.set_xlabel('S2 energy (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('# events', fontsize = 11)#ylabel
    ax.hist(df.S2e,
            range=(emin, emax),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.E.value, s2d.E.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 2)

    ax.set_xlabel(r'S2 width ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.S2w,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.W.value, s2d.W.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 3)

    ax.set_xlabel(r'Q (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.S2q,
            range=(0,1000),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.Q.value, s2d.Q.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 4)

    ax.set_xlabel(r'number SiPM',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.Nsipm,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.N.value, s2d.N.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 5)

    ax.set_xlabel(r' X (mm)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.X,
            range=(-200,200),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.X.value, s2d.X.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 6)
    ax.set_xlabel(r' Y (mm)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.Y,
            range=(-200,200),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.Y.value, s2d.Y.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    plt.tight_layout()
