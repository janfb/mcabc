import delfi.distribution as dd
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import pickle
import time

from delfi.generator import Default
from delfi.utils.viz import plot_pdf

from lfimodels.channelomics.ChannelSingle import ChannelSingle
from lfimodels.channelomics.ChannelSuper import ChannelSuper
from lfimodels.channelomics.ChannelStats import ChannelStats
from matplotlib import pyplot as plt

## GOAL of this script: generate k and na channel data and save to disk for later training
n_samples = 10
seed = 3
cython = True

# set groung truth
GT = {'k': np.array([9, 25, 0.02, 0.002]),
      'na': np.array([-35, 9, 0.182, 0.124, -50, -75, 5, -65, 6.2, 0.0091, 0.024])}

LP = {'k': ['qa','tha','Ra','Rb'],
      'na': ['tha','qa','Ra','Rb','thi1','thi2','qi','thinf','qinf','Rg','Rd']}

E_channel = {'k': -86.7, 'na': 50}
fact_inward = {'k': 1, 'na': -1}

gt_k = GT['k']
prior_lims_k = np.sort(np.concatenate((0.5 * gt_k.reshape(-1,1), 1.5 * gt_k.reshape(-1,1)), axis=1))

# as we use k as gt, the model is already set up..
mk = ChannelSingle(channel_type='k', n_params=len(gt_k), cython=cython, seed=seed)
pk = dd.Uniform(lower=prior_lims_k[:, 0], upper=prior_lims_k[:, 1], seed=seed)
sk = ChannelStats(channel_type='k', seed=seed)
gk = Default(model=mk, summary=sk, prior=pk)

# set up na model
gt_na = GT['na']
prior_lims_na = np.sort(np.concatenate((0.5 * gt_na.reshape(-1,1), 1.5 * gt_na.reshape(-1,1)), axis=1))
mna = ChannelSingle(channel_type='na', n_params=len(gt_na), cython=cython, seed=seed)
pna = dd.Uniform(lower=prior_lims_na[:, 0], upper=prior_lims_na[:,1], seed=seed)
sna = ChannelStats(channel_type='na', seed=seed)
gna = Default(model=mna, summary=sna, prior=pna)

# generate data
params_k, sx_k = gk.gen(n_samples=n_samples)
params_na, sx_na = gk.gen(n_samples=n_samples)

result_dict = dict(params_k=params_k, sx_k=sx_k, gt_k=gt_k, prior_lims_k=prior_lims_k,
                   params_na=params_na, sx_na=sx_na, gt_na=gt_na, prior_lims_na=prior_lims_na,
                   seed=seed, n_samples=n_samples, cython=cython)

# save data
folder = '../data'
filename = 'training_data_k_na_N{}seed{}'.format(n_samples, seed)
full_path_to_file = os.path.join(folder, filename + '.p')
with open(full_path_to_file, 'wb') as outfile:
    pickle.dump(result_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)