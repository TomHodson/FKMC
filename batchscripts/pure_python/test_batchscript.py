#!/usr/bin/env python
# coding: utf-8

# In[7]:


from munch import Munch

#give information to the dispatch script
batch_params = Munch(total_jobs = 5)

import os
job_id = int(os.getenv('JOB_ID', -1)) #this is the same for the whole array job
task_id = int(os.getenv('TASK_ID', 11)) #this is sequential for every task in the job

filename = f'{task_id}.npz'

print(f'job_id: {job_id}, task_id: {task_id}')

import numpy as np
np.savez_compressed(filename, np.arange(10), allow_pickle = True,)


# In[ ]:




