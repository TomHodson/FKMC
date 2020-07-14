
# coding: utf-8

# In[1]:


from pathlib import Path
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl

get_ipython().run_line_magic('matplotlib', 'inline')
np.seterr(all='warn')
textwidth = 6.268
mpl.rcParams['figure.dpi'] = 70
default_figargs = dict(figsize = (textwidth,textwidth))

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING) 


# In[3]:


this_run = Path('/data/users/tch14/completed_runs/88869')
print(f'looking in {this_run}')

datafiles = sorted([(int(f.stem), f) for f in this_run.iterdir() if f.name.endswith('npz') and not f.name == 'parameters.npz' and not f.name == 'aggregated.npz'])
jobs = np.array([j_id for j_id, f in datafiles])
#print([d.stem for i,d in datafiles])
print(f'job ids range from {min(jobs)} to {max(jobs)}')
print(f'missing jobs: {set(range(max(jobs))) - set(jobs)}')


# In[40]:





# In[ ]:


get_ipython().run_cell_magic('t', '', "data_slice = datafiles[:50]\nds = [np.load(f, allow_pickle = True) for i,f in data_slice]\n\nbetas = np.array([d['parameters'][()]['beta'] for d in ds])\nllogs = [d['logs'][()] for d in ds]\nbetas\n\nNs = ds[0]['Ns'][()]\nparameters = ds[0]['parameters'][()]\nMCMC_params = ds[0]['MCMC_params'][()]\n\nnp.savez_compressed(this_run / 'aggregated.npz',\n        Ns = Ns, Ts = 1/betas, parameters = parameters, MCMC_params = MCMC_params, logs = llogs, allow_pickle = True,\n        desc = ''\n        )")


# In[7]:


get_ipython().run_cell_magic('time', '', "data_agg = np.load(this_run / 'aggregated.npz', allow_pickle = True)\nlogs = data_agg['logs'][()]")


# In[11]:


list(ds[0].keys())


# In[17]:


log = logs[5]
print([key for key in dir(log) if not key.startswith('_')])


# In[13]:


from FKMC.general import running_mean
f, axes = plt.subplots(1,4, figsize = (20,5))

Mfs = np.array([[log.Mf_moments for log in logs] for logs in llogs])

for j in range(4):
    for i,N in enumerate(Ns):
            y = np.mean(Mfs[:, i, j], axis = -1)
            axes[j].scatter(1/betas, y)
axes[0].legend()
axes[2].set(xlim = (0.5,1.5), ylim = (0.8,1.0))


# In[113]:


def new_fs(new_betas, Fs, betas, fs, gs):
    #new_beta: the betas we want to interpolate to
    #betas: the set of temperatures at which the mcmc runs were performed
    #Fs.shape = (number of runs, number of step)
    #fs.shape = (number of runs), this need to be determined self consistently
    #final shapes: [new_beta index, runs over mcmc runs, runs over steps in an mcmc run, runs over runs again for innnermost sum]
    betas = betas[None, None, None, :]
    Fs3 = Fs[None, :, :]
    gs3 = gs[:]
    
    Fs4 = Fs[None, :, :, None]
    gs4 = gs[None, :, :, None]
    new_betas = new_betas[:, None, None]
    fs = fs[None, None, None, :]
    
    A = 1/gs3 * np.exp(- new_betas * Fs3) / np.sum(1/gs4 * np.exp(- betas * Fs4 + fs), axis = -1)
    expf = 1/Fs.shape[1] * np.sum(A, axis = (1,2))
    return - np.log(expf)
    
from scipy.optimize import fixed_point
def fit_fs(Fs, betas, fs_guess, gs):
    def func(fs): 
        if np.any(np.isnan(fs)): raise RuntimeError('The iteration hit a nan')
        return new_fs(betas, Fs, betas, fs, gs)
    fs = fixed_point(func, fs_guess, maxiter=1000)
    return fs

def FS_reweight(betas, beta, Fs, Os):
    exp_arg = -(betas[:, None] - beta) * Fs[None, :]
    #exp_arg -= np.max(exp_arg)
    
    boltz_factors = np.exp(exp_arg)
    return np.sum(boltz_factors * Os, axis = -1) / np.sum(boltz_factors, axis = -1)

def interp(Ts, Fs, Os, ax, **kwargs):
    for i in range(len(Ts))[1:-1]:
        new_Ts = np.linspace((Ts[i-1]+Ts[i])/2,(Ts[i+1]+Ts[i])/2, 20)

        reweighted_O = FS_reweight(betas = 1/new_Ts, beta = 1/Ts[i], Fs = Fs[i], Os = Os[i])

        ax.plot(new_Ts, reweighted_O, **kwargs)

def FS_multi_reweight(new_betas, Os, Fs, betas, fs_guess = None):
    #print(' '.join(f'{key} = {val.shape}' for key, val in locals().items()))
    print(Os.shape)
    gs = np.array([series_tau(x) for x in Os])
    fs_guess = fs_guess if not fs_guess is None else np.zeros_like(betas)
    fs = fit_fs(Fs, betas, fs_guess, gs)
    #print(fs)
    
    the_new_fs = new_fs(new_betas, Fs, betas, fs, gs)[:, None, None]
    new_betas = new_betas[:, None, None]
    gs3 = gs[:, None, None]
    Fs3 = Fs[None, :, :]

    
    betas = betas[None, None, None, :]
    fs = fs[None, None, None, :]
    gs4 = gs[None, None, None, :]
    Fs4 = Fs[None, :, :, None]

    
    O = 1/gs3 * Os * np.exp(- new_betas * Fs3 + the_new_fs) / np.sum(1/gs4 * np.exp(- betas * Fs4 + fs), axis = -1)
    
    return 1/Fs.shape[1] * np.sum(O, axis = (1,2)), fs[0,0,0,:]

def interp_multi(Ts, Fs, Os, ax, **kwargs):
    fs_guess = None
    for i in range(len(Ts))[2:-2]:
        try:
            ix = [i-2,i-1,i,i+1,i+2]
            new_Ts = np.linspace((Ts[i-1]+Ts[i])/2,(Ts[i]+Ts[i+1])/2, 20)

            reweighted_O, fs = FS_multi_reweight(new_betas = 1/new_Ts, betas = 1/Ts[ix], Fs = Fs[ix], Os = Os[ix])

            ax.plot(new_Ts, reweighted_O, **kwargs)
        except RuntimeError:
            pass


# In[114]:


from FKMC.general import running_mean
f, ax = plt.subplots(1,1, figsize = (20,10))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

Fs = np.array([[log.Ff + log.Ff for log in logs] for logs in llogs])

for i,N in enumerate(Ns):
        
            y = np.mean(Mfs[:, i, 2], axis = -1)
            yerr = np.std(Mfs[:, i, 2], axis = -1)
            
            ax.scatter(1/betas, y, color = colors[i], label = f'N = {N}')
            
            interp(1/betas, Fs[:, i], Mfs[:, i, 2], ax, color = colors[i])
            interp_multi(1/betas, Fs[:, i], Mfs[:, i, 2], ax, color = colors[i])
ax.legend()
#ax.set(xlim = (1,1.5), ylim = (0.8,1.01))


# In[104]:


from FKMC.general import running_mean

Ts = 1/betas
i = 30
T = Ts[i]

j = 5
N = Ns[j]
print(f'T = {T}, N = {N}')

y = Mfs[i, j, 2]
f ,axes = plt.subplots(1,3, figsize = (20,5))
axes[0].plot(y)
axes[1].plot(running_mean(y))


# ## Looking at autocorrelations calculated using a series

# In[105]:


from scipy.signal import correlate, convolve
def autocorrelation(X):
    N = X.shape[0]
    corr = correlate(X, X, mode = 'full')
    lagged = correlate(X, X, mode = 'full') / (N - np.abs(np.arange(1,2*N)-N))
    full = (lagged - np.mean(X)**2 ) / (np.mean(X**2) - np.mean(X)**2)
    return full[N:]

Cs = autocorrelation(y)
ts = np.arange(Cs.shape[0])

f ,axes = plt.subplots(1,3, figsize = (20,5))
axes[0].plot(ts, Cs)
axes[0].set(title = 'C(t)')
axes[0].plot(autocorrelation_2(y))

taus = 1/2 + np.cumsum(Cs)
axes[1].plot(ts, taus)
axes[1].set(title = '0.5 + sum_{0->i} C(i)', ylim = (min(taus), max(taus)))
axes[1].plot(ts,ts/5,'--')

crossings = np.argwhere(np.diff(np.sign(ts/5 - taus))).flatten()
axes[1].scatter(ts[crossings], taus[crossings])

def series_tau(X):
    Cs = autocorrelation(X)
    ts = np.arange(Cs.shape[0])
    taus = 1/2 + np.cumsum(Cs)
    
    crossings = np.argwhere(np.diff(np.sign(ts/5 - taus))).flatten()
    if len(crossings) != 1:
        return np.nan
    return taus[crossings]

print(series_tau(y))
    


# ## Looking at autocorrealtions calculated using binning

# In[106]:


def bin_error(X, binsize):
    s = X.std()
    binerr = X.reshape(-1, binsize).mean(axis = 1).std() * np.sqrt(binsize)
    return (binerr / s)**2 / 2
    
bins = [1,5,10,20,40,50,100,200,500,1000,2000,5000]
errs = np.array([bin_error(y, i) for i in bins])
plt.plot(bins, errs)


# In[85]:


f, ax = plt.subplots(1,1, figsize = (20,10))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

Fs = np.array([[log.Ff + log.Ff for log in logs] for logs in llogs])

for i,N in enumerate(Ns):
            taus = np.array([series_tau(x) for x in Mfs[:, i, 2]])
            
            ax.scatter(1/betas, taus, color = colors[i], label = f'N = {N}')
ax.legend()


# In[21]:


from FKMC.general import running_mean
f, axes = plt.subplots(1,4, figsize = (20,5))
x = np.arange(log.N_steps)

for j in range(4):
    for i,N in enumerate(Ns):
            log = logs[i]
            y = running_mean(log.Mf_moments[j])
            axes[j].plot(x, y)
axes[0].legend()


# In[34]:


from FKMC.general import running_mean
f, axes = plt.subplots(1,4, figsize = (20,5))
x = np.arange(log.N_steps)

for j in range(4):
    for i,N in enumerate(Ns):
            log = logs[i]
            y = log.Mf_moments[j]
            axes[j].hist(y, label = f'N = {N}')
axes[0].legend()


# In[ ]:


from FKMC.general import running_mean
f, axes = plt.subplots(1,4, figsize = (20,5))
x = np.arange(log.N_steps)

for j in range(4):
    for i,N in enumerate(Ns):
            log = logs[i]
            y = running_mean(log.Mf_moments[j])
            axes[j].plot(x, y)
axes[0].legend()


# In[24]:


from FKMC.general import running_mean
f, ax = plt.subplots(1,1, figsize = (20,5))
x = np.arange(log.N_steps)

for i,N in enumerate(Ns):
        log = logs[i]
        y = running_mean(np.mean(log.state, axis = -1))
        ax.plot(x, y, label = f'N={N}')
ax.legend()


# In[ ]:





# In[31]:


log = logs[-3]
N = Ns[-3]
print(f"System size = {N}, steps = {log.N_steps}, thin = {d['MCMC_params'][()]['thin']}")
print(f'mean(M^2) = {np.mean(log.Mf_moments[2])}')
print(f'std(M^2) = {np.std(log.Mf_moments[2])}')
      


# In[ ]:


from zipfile import BadZipfile
zips = []
data = []
for n,file in datafiles:
    try:
        z = np.load(file, allow_pickle = True)
        d = z['logs'][()]
        zips.append(z)
        data.append(d)
    except (BadZipfile, OSError):
        print(f'{file.name} is corrupted')


# In[48]:


f'{sum([log.time for log in data]) / (60*60):.0f} hours total'


# In[50]:


Ts = parameters['Ts']
assert(len(data) == len(Ts))

f, ax = plt.subplots(1, figsize = (20,5))
for T, d in zip(Ts,data):
    F = d.Ff + d.Fc
    #print(f'T = {T}, F.mean() = {F.mean()}')
    ax.hist(F, label = T, alpha = 0.5)
#ax.legend()


# In[12]:


f, ax = plt.subplots(1)
for T, d in zip(Ts,data):
    ax.hist(d.Mf_moments[2], label = T)
#ax.legend()


# In[51]:


y = np.array([log.Mf_moments[2].mean() for log in data])
yerr = np.array([log.Mf_moments[2].std() for log in data]) 
x = Ts

f, ax = plt.subplots(1)
ax.errorbar(x, y, yerr = yerr)


# In[68]:


def new_fs(new_betas, Fs, betas, fs):
    #new_beta: the betas we want to interpolate to
    #betas: the set of temperatures at which the mcmc runs were performed
    #Fs.shape = (number of runs, number of step)
    #fs.shape = (number of runs), this need to be determined self consistently
    #final shapes: [new_beta index, runs over mcmc runs, runs over steps in an mcmc run, runs over runs again for innnermost sum]
    betas = betas[None, None, None, :]
    Fs3 = Fs[None, :, :]
    Fs4 = Fs[None, :, :, None]
    new_betas = new_betas[:, None, None]
    fs = fs[None, None, None, :]
    
    A = np.exp(- new_betas * Fs3) / np.sum(np.exp(- betas * Fs4 + fs), axis = -1)
    expf = 1/Fs.shape[1] * np.sum(A, axis = (1,2))
    return - np.log(expf)
    
from scipy.optimize import fixed_point
def fit_fs(Fs, betas, fs_guess):
    def func(fs): 
        if np.any(np.isnan(fs)): raise RuntimeError('The iteration hit a nan')
        return new_fs(betas, Fs, betas, fs)
    fs = fixed_point(func, fs_guess, maxiter=1000)
    return fs
    

Fs = np.array([log.Ff + log.Fc for log in data])
O = np.array([log.Mf_moments[2] for log in data])
Ts = parameters['Ts']

y = O.mean(axis = -1)
yerr = O.std(axis = -1)



f, ax = plt.subplots(1, figsize = (15,5))
ax.scatter(Ts, y, marker = 'o')
ax.set(
    title = 'FS reweight single point',
    xlim = (0.5,4),
    )
        
interp(Ts, Fs, Os, ax)

def FS_multi_reweight(new_betas, Os, Fs, betas, fs_guess = None):
    #print(' '.join(f'{key} = {val.shape}' for key, val in locals().items()))
    fs_guess = fs_guess if not fs_guess is None else np.zeros_like(betas)
    fs = fit_fs(Fs, betas, fs_guess)
    print(fs)
    
    the_new_fs = new_fs(new_betas, Fs, betas, fs)[:, None, None]
    new_betas = new_betas[:, None, None]
    Fs3 = Fs[None, :, :]
    
    betas = betas[None, None, None, :]
    fs = fs[None, None, None, :]
    Fs4 = Fs[None, :, :, None]
    
    O = Os * np.exp(- new_betas * Fs3 + the_new_fs) / np.sum(np.exp(- betas * Fs4 + fs), axis = -1)
    
    return 1/Fs.shape[1] * np.sum(O, axis = (1,2)), fs[0,0,0,:]

def interp_multi(Ts, Fs, Os, ax):
    fs_guess = None
    for i in range(len(Ts))[3:-5]:
        try:
            ix = [i-1,i,i+1]
            new_Ts = np.linspace((Ts[i-1]+Ts[i])/2,(Ts[i]+Ts[i+1])/2, 20)

            reweighted_O, fs = FS_multi_reweight(new_betas = 1/new_Ts, betas = 1/Ts[ix], Fs = Fs[ix], Os = O[ix])

            ax.plot(new_Ts, reweighted_O, color = 'r')
        except RuntimeError:
            pass
                             
interp_multi(Ts, Fs, Os, ax)


# In[209]:


def FS_reweight(betas, beta, Fs, Os):
    exp_arg = -(betas[:, None] - beta) * Fs[None, :]
    #exp_arg -= np.max(exp_arg)
    
    boltz_factors = np.exp(exp_arg)
    return np.sum(boltz_factors * Os, axis = -1) / np.sum(boltz_factors, axis = -1)

def interp(Ts, Fs, Os, ax):
    for i in range(len(Ts))[1:-1]:
        new_Ts = np.linspace((Ts[i-1]+Ts[i])/2,(Ts[i+1]+Ts[i])/2, 20)

        reweighted_O = FS_reweight(betas = 1/new_Ts, beta = 1/Ts[i], Fs = Fs[i], Os = O[i])

        ax.plot(new_Ts, reweighted_O, color = 'g')

Fs = np.array([log.Ff + log.Fc for log in data])
O = np.array([log.Mf_moments[2] for log in data])

y = O.mean(axis = -1)
yerr = O.std(axis = -1)

Ts = parameters['Ts']

f, ax = plt.subplots(1, figsize = (15,5))
ax.scatter(Ts, y, marker = 'o')
ax.set(
    title = 'FS reweight single point',
    #xlim = (1,3),
    )
        
interp(Ts, Fs, Os, ax)


# In[208]:


Fs = np.array([log.Ff + log.Fc for log in data])
O = (Fs - Fs.mean(axis=-1)[:, None])**2

y = O.mean(axis = -1)
yerr = O.std(axis = -1)

Ts = parameters['Ts']

f, ax = plt.subplots(1, figsize = (15,5))
ax.scatter(Ts, y, marker = 'o')
ax.set(
    title = 'FS reweight single point',
    #xlim = (1,3),
    )
        
interp(Ts, Fs, Os, ax)


# In[187]:


y = np.array([log.Mf_moments[2].mean() for log in data])
yerr = np.array([log.Mf_moments[2].std() for log in data]) 
x = Ts = parameters['Ts']

f, ax = plt.subplots(1, figsize = (15,5))
ax.scatter(x, y, marker = 'o')
ax.set(
    title = 'FS reweight',
    xlim = (3,4),
    ylim = (0,0.02)
    )

def FS_reweight_two_point(betas, beta, Fs, Os):
    beta, Fs, Os = np.broadcast_arrays(beta[:, None], Fs, Os)
    #beta = beta[:, None] * np.ones(Fs.shape)[None, :]
    beta, Fs, Os = np.array([beta, Fs, Os]).reshape(3,-1)
    
    exp_arg = -(betas[:, None] - beta[None, :]) * Fs[None, :]
    #exp_arg -= np.max(exp_arg)
    
    boltz_factors = np.exp(exp_arg)
    return np.sum(boltz_factors * Os, axis = -1) / np.sum(boltz_factors, axis = -1)

for i in range(len(data))[1:-1]:
    new_Ts = np.linspace(x[i-1],x[i], 20)
    #new_Ts = np.array([x[i],])

    betas = 1/new_Ts
    Fs = data[i].Ff + data[i].Fc
    Os = data[i].Mf_moments[2]

    reweighted_O = FS_reweight(betas, 1/x[i], Fs, Os)
    ax.plot(new_Ts, reweighted_O, color = 'r')
print('single point rewighting done')
    
for i in range(len(data))[1:-1]:
    new_Ts = np.linspace(x[i-1],x[i], 10)
    betas = 1/new_Ts
    
    beta = np.array([1/Ts[i-1], 1/Ts[i]])
    Fs = np.array([data[i-1].Ff + data[i-1].Fc, data[i].Ff + data[i].Fc])
    Os = np.array([data[i-1].Mf_moments[2], data[i].Mf_moments[2]])

    reweighted_O = FS_reweight_two_point(betas, beta, Fs, Os)
    ax.plot(new_Ts, reweighted_O, '--g')


# In[ ]:





# In[122]:


from FKMC.general import running_mean
f, axes = plt.subplots(1,4, figsize = (20,5))
for j in range(4):
    for i in range(len(Ts)):
        T = Ts[i]
        log = data[i]
        y = running_mean(log.Mf_moments[j+1])
        axes[j].plot(y)
ax.legend()


# In[91]:


from FKMC.general import running_mean
f, ax = plt.subplots(1)
for i in range(len(Ts)):
    T = Ts[i]
    log = data[i]
    y = running_mean(log.Ff + log.Fc)
    x = np.arange(len(y))
    ax.errorbar(x, y)
ax.legend()


# In[127]:


i = 7
f, ax = plt.subplots(1)
T = Ts[i]
print(f'T = {T}')
log = data[i]

#average site occupations
A = 2*(np.arange(log.N_sites) % 2) - 1
plt.plot(log.state.mean(axis = 0))


# In[21]:


n_by_ns = np.zeros((50,256,256))
corrs = np.zeros((50,256,256))
cs = np.zeros((50,256))

A = 2*(np.arange(log.N_sites) % 2) - 1
for i in range(50):
    log = data[i]
    t = A * (2*log.state - 1)
    n_by_ns[i] = np.mean(t[:, :, None] * t[:, None, :], axis = 0)

    for j in range(log.N_sites):
        corrs[i,j] = np.roll(n_by_ns[i,j], shift = -j, axis = 0)

    cs[i] = corrs[i].mean(axis = 0)


# In[23]:


Is = np.arange(15)
f, axes = plt.subplots(3,len(Is), figsize = (,10))
norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
cmap = plt.cm.get_cmap("winter")
for ax, i in zip(axes.T, Is):
    T = Ts[i]
    log = data[i]

    ax[2].set(title = f'T = {T:.2f}')
    ax[0].imshow(n_by_ns[i], cmap = cmap, norm = norm)
    ax[1].imshow(corrs[i], cmap = cmap, norm = norm)
    ax[2].plot(cs[i])


# In[ ]:




