import numpy as np
import scipy
np.seterr(all='warn')

from pathlib import Path
import os
from munch import Munch
import pickle
import logging
from itertools import count
from time import time

from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING) 

mpl.rcParams['figure.dpi'] = 70

print('Imported pathlib::Path, os, munch::Munch, pickle, logging, itertools::count, matplotlib::colors')
print('Names are pyplot = plt, matplotlib = mpl, numpy = np')

