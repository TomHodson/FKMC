import numpy as np
import scipy

from itertools import count
from munch import Munch
import logging
import multiprocessing as mp

from FKMC.general import index_histogram_array, sort_IPRs, smooth, shapes, normalise_IPR
from FKMC.import_funcs import shape_hints, timefmt

from FKMC.import_funcs import  mean_over_MCMC, IPRandDOS, extract, get_data_funcmap_chain_ext, extractStates
from FKMC.import_funcs import incremental_get_data_funcmap_chain_ext, incremental_load

from scipy.stats import sem
from FKMC.general import scaling_dimension

def interpolate_IPR(E_bins, unsmoothed_DOS, IPR, dIPR):
    newshape = (IPR.size // IPR.shape[-1], IPR.shape[-1])
    _DOS = unsmoothed_DOS.reshape(newshape)
    _IPR = IPR.reshape(newshape)
    _dIPR = dIPR.reshape(newshape)
    
    for i, DOS, I, dI in zip(count(), _DOS, _IPR, _dIPR):
        ei = DOS > 0
        if any(ei):
            _I = I[ei]
            _dI = dI[ei]
            xI = E_bins[1:][ei]

            _IPR[i] = np.interp(E_bins[1:], xI, _I)
            _dIPR[i] = np.interp(E_bins[1:], xI, _dI)
        else:
            _IPR[i] = E_bins[1:] * np.NaN
            _dIPR[i] = E_bins[1:] * np.NaN

def fit_errors(X, Y, dY):
    try:
        (m, c), cov = np.ma.polyfit(X, Y, deg = 1, cov=True, w = 1 / dY)
        dm, dc = np.sqrt(np.diag(cov))
        return m, c, dm, dc
    except np.linalg.LinAlgError:
        return np.NaN, np.NaN, np.NaN, np.NaN

def fit_no_errors(X, Y):
    try:
        (m, c), cov = np.ma.polyfit(X, Y, deg = 1, cov=True)
        dm, dc = np.sqrt(np.diag(cov))
        return m, c, dm, dc
    except np.linalg.LinAlgError:
        return np.NaN, np.NaN, np.NaN, np.NaN

def scaling_dimension_multidim(Ns, IPR, dIPR, use_true_errors = True):
    original_shape = IPR.shape
    newshape = (IPR.shape[0], IPR.size // IPR.shape[0])
    finalshape = IPR.shape[1:]
    IPR = IPR.reshape(newshape)
    dIPR = dIPR.reshape(newshape)
    print(original_shape, newshape, finalshape)
    
    Y = np.log(IPR).T
    dY = dIPR.T / IPR.T #take the maximum error across the energy spectrum because we can't do it individually
    #set a minimum 5% error
    dY = np.maximum(dY, 5/100)
    X = np.broadcast_to(np.log(Ns), Y.shape)
    
    with mp.Pool(16) as pool:
        if use_true_errors:
            args = np.stack([X, Y, dY], axis = 1)
            fit = fit_errors
        else:
            args = np.stack([X, Y, dY], axis = 1)
            fit = fit_no_errors
        
        print(args.shape)
        m, c, dm, dc = np.array(pool.starmap(fit, args, chunksize = 1000)).T

    return m.reshape(finalshape), c.reshape(finalshape), dm.reshape(finalshape), dc.reshape(finalshape)

from FKMC.general import scaling_dimension

def interpolate_and_smooth(o):
    interpolate_IPR(o.E_bins, unsmoothed_DOS=o.DOS, IPR=o.IPR, dIPR=o.dIPR)

    o.dIPR = sem(o.IPR, axis = 1)
    o.IPR = np.mean(o.IPR, axis = 1)
    o.dDOS = sem(o.DOS, axis = 1)
    o.DOS = np.mean(o.DOS, axis = 1)

    o.IPR = smooth(o.IPR, scale = 0.5, axis = -1)
    o.dIPR = smooth(o.dIPR, scale = 0.5, axis = -1)
    o.DOS = smooth(o.DOS, scale = 0.5, axis = -1)

    try:
        o.m, o.c, o.dm, o.dc = scaling_dimension(o.Ns, o.IPR, o.dIPR, use_true_errors = True)
    except:
        print('Scaling dimension fit failed on at least one value, falling back to loop')
        o.m, o.c, o.dm, o.dc = scaling_dimension_multidim(o.Ns, o.IPR, o.dIPR, use_true_errors = True)
    
    return o

functions = [
    mean_over_MCMC('Mf_moments', N_error_bins = 10),
    IPRandDOS(),
    extractStates(),
]
o1 = incremental_load(folder = '~/HPC_data/gap_open_U=2', functions = functions, force_reload = True)
print(o1.Ts)

functions = [
    mean_over_MCMC('Mf_moments', N_error_bins = 10),
    IPRandDOS(),
]
o2 = incremental_load(folder = '~/HPC_data/gap_opening_U=2_low_temp', functions = functions, force_reload = False)

if len(o1.IPR.shape) == 4:
    assert(all(o1.Ts[:10] ==  o2.Ts)) #assert that 02 just fills in the lower T data of o1
    for name in ['IPR', 'DOS']:
        o1[name][:, :, :10, :] = o2[name]
    

o1 = interpolate_and_smooth(o1)

with open('/home/tch14/HPC_data/pickled_data/gap_opening.pickle', 'wb') as file:
    pickle.dump(o1, file)