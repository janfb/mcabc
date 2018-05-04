import numpy as np
import os
import pickle
import sys
import tempfile
import time
import tqdm

from delfi.utils.viz import plot_pdf

from lfimodels.channelomics.ChannelSingle import ChannelSingle
from lfimodels.channelomics.ChannelStats import ChannelStats

from pyabc import (ABCSMC, RV,
                   PercentileDistanceFunction, DistanceFunction, sampler)
from pyabc import Distribution as abcDis

sys.path.append('../../')
from model_comparison.utils import *
from model_comparison.mdns import *

GT = {'kd': np.array([[4, -63, 0.032, 15, 5, 0.5, 10, 40]]),
      'kslow': np.array([[1, 35, 10, 3.3, 20]])}

LP = {'kd': ['power',r'$V_T$',r'$R_{\alpha}$',r'$th_{\alpha}$', r'$q_{\alpha}$', r'$R_{\beta}$', r'$th_{\beta}$',
             r'$q_{\beta}$'],
      'kslow': ['power', r'$V_T$', r'$q_p$', r'$R_{\tau}$', r'$q_{\tau}$']}

E_channel = {'kd': -90.0, 'kslow': -90.0}
fact_inward = {'kd': 1, 'kslow': 1}

prior_lims_kd = np.sort(np.concatenate((0.9 * GT['kd'].reshape(-1, 1), 1.2 * GT['kd'].reshape(-1, 1)), axis=1))
prior_lims_ks = np.sort(np.concatenate((0.9 * GT['kslow'].reshape(-1, 1), 1.2 * GT['kslow'].reshape(-1, 1)), axis=1))

cython = True
seed = 2

m_obs = ChannelSingle(channel_type='kd', n_params=8, cython=cython)
s = ChannelStats(channel_type='kd')

xo = m_obs.gen(GT['kd'].reshape(1,-1))
sxo = s.calc(xo[0])

mkd = ChannelSingle(channel_type='kd', n_params=8, cython=cython, seed=seed)
skd = ChannelStats(channel_type='kd', seed=seed)

mks = ChannelSingle(channel_type='kslow', n_params=5, cython=cython, seed=seed)
sks = ChannelStats(channel_type='kslow', seed=seed)


# Define models oin pyabc style
def model_1(parameters):
    params = np.array([parameters.p1, parameters.p2, parameters.p3, parameters.p4,
                       parameters.p5, parameters.p6, parameters.p7, parameters.p8])
    x = mkd.gen(params.reshape(1, -1))
    sx = skd.calc(x[0])
    return {'y': sx}


def model_2(parameters):
    params = np.array([parameters.p1, parameters.p2, parameters.p3, parameters.p4, parameters.p5])
    x = mks.gen(params.reshape(1, -1))
    sx = sks.calc(x[0])
    return {'y': sx}


# priors
prior_dict_kd = dict()
for i in range(8):
    prior_dict_kd['p{}'.format(i + 1)] = dict(type='uniform',
                                              kwargs=dict(loc=prior_lims_kd[i, 0],
                                                          scale=prior_lims_kd[i, 1] - prior_lims_kd[i, 0]))

prior1 = abcDis.from_dictionary_of_dictionaries(prior_dict_kd)

prior_dict_ks = dict()
for i in range(5):
    prior_dict_ks['p{}'.format(i + 1)] = dict(type='uniform',
                                              kwargs=dict(loc=prior_lims_ks[i, 0],
                                                          scale=prior_lims_ks[i, 1] - prior_lims_ks[i, 0]))

prior2 = abcDis.from_dictionary_of_dictionaries(prior_dict_ks)

models = [model_1, model_2]
parameter_priors = [prior1, prior2]


class MyDist(DistanceFunction):

    def __call__(self, x, y):
        return np.power(x['y'] - y['y'], 2).mean()

fn = 'training_data_kd_ks_N100seed1.p'
with open(os.path.join('../data', fn), 'rb') as f:
    dtest = pickle.load(f)
dtest.keys()

sx_test_ks = dtest['sx_ks']
sx_test_kd = dtest['sx_kd']

fn = 'learned_posteriors_pospischil_ntrain192962.p'
with open(os.path.join('../data', fn), 'rb') as f:
    dpost = pickle.load(f)['model_idx_posterior']
dpost.keys()

upto = 50
test_set = np.vstack((sx_test_kd[:upto, ], sx_test_ks[:upto, ]))
mtest = np.hstack((np.zeros(upto), np.ones(upto))).astype(int).tolist()
ntest = test_set.shape[0]
phat_smc = np.zeros((ntest, 2))
phat_mdn = np.zeros((ntest, 2))

# get mdn
model_mdn = dpost['model_idx_mdn']
data_norm = dpost['data_norm']

n_rounds = 5
n_simulations = 0

tic = time.time()
for ii in tqdm.tqdm(range(ntest)):
    sxo = test_set[ii, ]

    # We plug all the ABC options together
    abc = ABCSMC(
        models, parameter_priors, MyDist())

    # and we define where to store the results
    db_path = ("sqlite:///" +
               os.path.join(tempfile.gettempdir(), "test.db"))
    abc_id = abc.new(db_path, {"y": sxo})

    history = abc.run(minimum_epsilon=1e-7, max_nr_populations=n_rounds)
    model_probabilities = history.get_model_probabilities().as_matrix()
    print(model_probabilities)
    print(history.total_nr_simulations)
    n_simulations += history.total_nr_simulations

    try:
        phat_smc[ii, 0] = model_probabilities[model_probabilities.shape[0] - 1, 0]
    except IndexError:
        phat_smc[ii, 0] = model_probabilities[model_probabilities.shape[0] - 1]
    phat_smc[ii, 1] = 1 - phat_smc[ii, 0]

time_smc = time.time() - tic

tic = time.time()
for ii in tqdm.tqdm(range(ntest)):
    sxo = test_set[ii,]

    # predict with mdn
    sxo_zt, _ = normalize(sxo, data_norm)
    phat_mdn[ii,] = model_mdn.predict(sxo_zt.reshape(1, -1))

time_de = time.time()


d = dict(mtest=mtest, sx_test=test_set, ppoi_hat=phat_mdn[:, 0], ppoi_smc=phat_smc[:, 0], data_norm=data_norm,
         time_de=time_de, time_smc=time_smc, n_simulations=n_simulations)

time_stamp = time.strftime('%Y%m%d%H%M_')

fn = time_stamp + '_modelposterior_comparison_channels_ntest{}.p'.format(ntest)
with open(os.path.join('../data', fn), 'wb') as outfile:
    pickle.dump(d, outfile, protocol=pickle.HIGHEST_PROTOCOL)
