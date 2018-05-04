import delfi.distribution as dd
import numpy as np
import os
import pickle
import time

from lfimodels.channelomics.ChannelSingle import ChannelSingle
from lfimodels.channelomics.ChannelStats import ChannelStats
from lfimodels.channelomics.ChannelMPGenerator import ChannelMPGenerator


# GOAL of this script: generate k and na channel data and save to disk for later training
n_samples = int(5 * 1e5)
seed = 1
cython = True

start = time.time()
# set groung truth
GT = {'kd': np.array([[4, -63, 0.032, 15, 5, 0.5, 10, 40]]),
      'kslow': np.array([[1, 35, 10, 3.3, 20]])}

LP = {'kd': ['power',r'$V_T$',r'$R_{\alpha}$',r'$th_{\alpha}$', r'$q_{\alpha}$', r'$R_{\beta}$', r'$th_{\beta}$',
             r'$q_{\beta}$'],
      'kslow': ['power', r'$V_T$', r'$q_p$', r'$R_{\tau}$', r'$q_{\tau}$']}

gt_k = GT['kd']
prior_lims_k = np.sort(np.concatenate((0.7 * gt_k.reshape(-1, 1), 1.2 * gt_k.reshape(-1, 1)), axis=1))

n_workers = 30

# as we use k as gt, the model is already set up..
model_seeds = np.arange(1, n_workers + 1)
mks = [ChannelSingle(channel_type='kd', n_params=len(gt_k), cython=cython, seed=model_seeds[i]) for i in range(n_workers)]
pk = dd.Uniform(lower=prior_lims_k[:, 0], upper=prior_lims_k[:, 1], seed=seed)
sk = ChannelStats(channel_type='kd', seed=seed)
# gk = Default(model=mk, summary=sk, prior=pk, seed=seed)
gk = ChannelMPGenerator(models=mks, summary=sk, prior=pk, seed=seed)

# set up kslow model
gt_ks = GT['kslow']
prior_lims_ks = np.sort(np.concatenate((0.7 * gt_ks.reshape(-1, 1), 1.2 * gt_ks.reshape(-1, 1)), axis=1))

model_seeds = np.arange(n_workers, 2 * n_workers)
mnas = [ChannelSingle(channel_type='kslow', n_params=len(gt_ks), cython=cython, seed=model_seeds[i]) for i in range(n_workers)]
pna = dd.Uniform(lower=prior_lims_ks[:, 0], upper=prior_lims_ks[:, 1], seed=seed)
sna = ChannelStats(channel_type='kslow', seed=seed)
# gna = Default(model=mna, summary=sna, prior=pna, seed=seed)
gks = ChannelMPGenerator(models=mnas, summary=sna, prior=pna, seed=seed)


# generate data
params_kd, sx_kd = gk.gen(n_samples=n_samples)
params_ks, sx_ks = gks.gen(n_samples=n_samples)

result_dict = dict(params_kd=params_kd, sx_kd=sx_kd, gt_kd=gt_k, prior_lims_kd=prior_lims_k,
                   params_ks=params_ks, sx_ks=sx_ks, gt_ks=gt_ks, prior_lims_ls=prior_lims_ks,
                   seed=seed, n_samples=n_samples, cython=cython)

# save data
folder = '../data'
time_stamp = time.strftime('%Y%m%d%H%M_')
filename = time_stamp + '_training_data_kd_ks_N{}seed{}_prior0912'.format(n_samples, seed)
full_path_to_file = os.path.join(folder, filename + '.p')
with open(full_path_to_file, 'wb') as outfile:
    pickle.dump(result_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

print(time.time() - start)
