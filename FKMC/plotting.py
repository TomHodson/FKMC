from FKMC.general import scaling_dimension
import numpy as np
import scipy
from munch import Munch
from itertools import count
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl

def spread(ax, X, Y, dY, alpha = 0.3, **kwargs): 
    'plot Y against X but with a solid error spread of dY, return line, error_bar'
    l, = ax.plot(X, Y, **kwargs)
    fill = ax.fill_between(X, Y+dY, Y-dY, alpha = alpha, color = l.get_color())
    return l, fill 

def plot_scaling_dimension(original_data, filter_indices = None, spread_IPR = False, cutoffs = [0,], DOS_cutoff = 0.01, axes = None, Nmask = slice(None), colors = None):
    #make some default axes if necessary
    if axes is None:
        f, axes = plt.subplots(3,1, figsize = (15,10), 
                       sharex = 'col', sharey = 'row',
                      gridspec_kw = dict(wspace = 0, hspace = 0),
                      )
    else: f = None
    
    #if the data has shape like [Ns, Ts] and you wanna select a specific T
    #set filter_indices = (slice(None), T_i)
    if filter_indices:
        o = Munch()
        for obs in ['DOS', 'IPR', 'dIPR', 'dDOS']:
            o[obs] = original_data[obs][filter_indices]
        for obs in ['Ns', 'E_bins', 'parameters', 'MCMC_params']:
            o[obs] = original_data[obs]
    else:
        o = original_data
    
    cmap=plt.get_cmap('tab10')
    colors = colors or cmap(np.linspace(0,0.8,len(o.Ns)))
    
    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    Eidx = o.DOS[-1] < DOS_cutoff #region of insufficient data
    Erange = o.E_bins[1:][~Eidx] / o.parameters.U
    Emin, Emax = np.min(Erange), np.max(Erange)
    
    for i,N in zip(count(), o.Ns):
        if N not in o.Ns[Nmask]: continue
            
        IPR = o.IPR[i].copy()
        IPR[Eidx] = np.NaN
        #compute the bounds of the IPR that gets plotted
        bw = o.E_bins[1] - o.E_bins[0]
        #print(f'sum = {o.DOS[i].sum() * bw}')
        assert(abs(o.DOS[i].sum() * bw - 1) < 0.01)
        
        
        axes[0].plot(o.E_bins[1:] / o.parameters.U, o.DOS[i], label = f'N = {N}', color = colors[i])
        if spread_IPR:
            spread(axes[1], o.E_bins[1:] / o.parameters.U, IPR, o.dIPR[i], color = colors[i])
        else:    
            axes[1].plot(o.E_bins[1:] / o.parameters.U, IPR, color = colors[i])

        axes[0].set(xlim = (-1.5, 1.5))

    
    try:
        for cutoff in cutoffs:
            idx = (o.Ns >= cutoff)
            m, c, dm, dc = scaling_dimension(o.Ns[idx], o.IPR[idx], o.dIPR[idx], use_true_errors = True)
            m[Eidx] = np.NaN
            spread(axes[2], o.E_bins[1:] / o.parameters.U, -m, dm, label = f'{cutoff} <= N < {max(o.Ns)}')

    except scipy.linalg.LinAlgError:
        pass


    xlim = axes[2].get_xlim()
    axes[2].hlines([0,1], xmin = xlim[0], xmax = xlim[1], linestyles = 'dashed')
    
    return f, axes

    