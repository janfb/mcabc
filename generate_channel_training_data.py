import delfi.distribution as dd
import matplotlib as mpl
import numpy as np
import pandas as pd
import time

from delfi.generator import Default
from delfi.utils.viz import plot_pdf

from lfimodels.channelomics.ChannelSingle import ChannelSingle
from lfimodels.channelomics.ChannelSuper import ChannelSuper
from lfimodels.channelomics.ChannelStats import ChannelStats
from matplotlib import pyplot as plt

# set groung truth
GT = {'k': np.array([9, 25, 0.02, 0.002]),
      'na': np.array([-35, 9, 0.182, 0.124, -50, -75, 5, -65, 6.2, 0.0091, 0.024])}

LP = {'k': ['qa','tha','Ra','Rb'],
      'na': ['tha','qa','Ra','Rb','thi1','thi2','qi','thinf','qinf','Rg','Rd']}

E_channel = {'k': -86.7, 'na': 50}
fact_inward = {'k': 1, 'na': -1}

channel_type = 'k'
gt = GT[channel_type]
cython = True
third_exp_model = True

n_params = len(gt)
labels_params = LP[channel_type]
prior_lims = np.sort(np.concatenate((0.5 * gt.reshape(-1,1), 1.5 * gt.reshape(-1,1)), axis=1))

m = ChannelSuper(channel_type=channel_type, third_exp_model=third_exp_model, cython=cython)
p = dd.Uniform(lower=prior_lims[:,0], upper=prior_lims[:,1])
s = ChannelStats(channel_type=channel_type)

# generate observed data
n_params_obs = len(gt)
m_obs = ChannelSingle(channel_type=channel_type, n_params=n_params_obs, cython=cython)
xo = m_obs.gen(gt.reshape(1,-1))
xo_stats = s.calc(xo[0])

seed = 3
gt_k = GT['k']
# as we use k as gt, the model is already set up..
mk = m
pk = p
sk = s

# set up na model
gt_na = GT['na']
prior_lims_na = np.sort(np.concatenate((0.5 * gt_na.reshape(-1,1), 1.5 * gt_na.reshape(-1,1)), axis=1))
mna = ChannelSuper(channel_type='na', third_exp_model=third_exp_model, cython=cython)
pna = dd.Uniform(lower=prior_lims_na[:,0], upper=prior_lims_na[:,1])
sna = ChannelStats(channel_type='na')

# generate params
n_samples = 100
params_k = pk.gen(n_samples=n_samples)
params_na = pna.gen(n_samples=n_samples)

# simulate
x_k = mk.gen(params_list=params_k)
x_na = mk.gen(params_list=params_na)
