
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


# In[2]:


this_run = Path('/data/users/tch14/slurm_runs/89173')
print(f'looking in {this_run}')

datafiles = sorted([(int(f.stem), f) for f in this_run.iterdir() if f.name.endswith('npz') and not f.name == 'parameters.npz'])
jobs = np.array([j_id for j_id, f in datafiles])
#print([d.stem for i,d in datafiles])
print(f'job ids range from {min(jobs)} to {max(jobs)}')
print(f'missing jobs: {set(range(max(jobs))) - set(jobs)}')


# In[40]:





# In[8]:


get_ipython().run_cell_magic('time', '', "print(f'Loading the files from disk lazily')\nds = [np.load(f, allow_pickle = True) for i,f in datafiles[:20]]")


# In[9]:


get_ipython().run_cell_magic('time', '', "betas = np.array([d['parameters'][()]['beta'] for d in ds])\nllogs = [d['logs'][()] for d in ds]\nbetas\n\nNs = ds[0]['Ns'][()]")


# In[6]:


1/betas


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

