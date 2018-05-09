import pickle
import sys
import time
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

# load samples
fn = '../data/201804280631__training_data_kd_ks_N1e6seed7.p'

with open(fn, 'rb') as f:
    d = pickle.load(f)

sx_ks = d['sx_ks']
sx_kd = d['sx_kd']

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

upto = 2
test_set = np.vstack((sx_test_kd[:upto, ], sx_test_ks[:upto, ]))
mtest = np.hstack((np.zeros(sx_test_kd[:upto, ].shape[0]),
                    np.ones(sx_test_ks[:upto, ].shape[0]))).astype(int).tolist()
ntest = test_set.shape[0]
# get mdn
model_mdn = dpost['model_idx_mdn']
data_norm = dpost['data_norm']

tic = time.time()

priors = np.arange(0.1, 1., 0.1)
n_priors = priors.shape[0]

phat_rej = np.zeros((n_priors, ntest, 2))
phat_mdn = np.zeros((ntest, 2))

for iprior, pkd in enumerate(priors):
    for ii in tqdm.tqdm(range(ntest)):
        sxo = test_set[ii, ]

        accepted_mi, data_set_indices, differences = rejection_abc_from_stats(sxo, [sx_kd, sx_ks], [pkd, 1 - pkd],
                                                                              niter=100000, verbose=False, eps=5e-6)

        phat_rej[iprior, ii, 1] = np.mean(accepted_mi)
        phat_rej[iprior, ii, 0] = 1 - phat_rej[iprior, ii, 1]

time_smc = time.time() - tic

tic = time.time()
for ii in tqdm.tqdm(range(ntest)):
    sxo = test_set[ii,]

    # predict with mdn
    sxo_zt, _ = normalize(sxo, data_norm)
    phat_mdn[ii, ] = model_mdn.predict(sxo_zt.reshape(1, -1))

time_de = time.time()


d = dict(mtest=mtest, sx_test=test_set, ppoi_hat=phat_mdn[:, 0], ppoi_rej=phat_rej[:, :, 0], data_norm=data_norm,
         time_de=time_de, time_smc=time_smc)

time_stamp = time.strftime('%Y%m%d%H%M_')

fn = time_stamp + '_modelposterior_comparison_rejection_sampling_channels_ntest{}_with_priorcheck.p'.format(ntest)
with open(os.path.join('../data', fn), 'wb') as outfile:
    pickle.dump(d, outfile, protocol=pickle.HIGHEST_PROTOCOL)
