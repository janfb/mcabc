import pickle
import sys
import scipy.stats


sys.path.append('../../')
from model_comparison.mdns import *
from model_comparison.models import NegativeBinomialModel

# set the seed for generating new test data
seed = 5
np.random.seed(seed)

# priors
sample_size = 100
ntrain = int(100000)
k2, k3 = 20., 2.
theta2, theta3 = 1., 2.

prior_k = scipy.stats.gamma(a=k2, scale=theta2)
prior_theta = scipy.stats.gamma(a=k3, scale=theta3)

model_nb = NegativeBinomialModel(sample_size=sample_size, seed=seed)

# generate training data
params_train = np.vstack((prior_k.rvs(size=ntrain), prior_theta.rvs(size=ntrain))).T
x_train = model_nb.gen(params_train)

sx_train = calculate_stats_toy_examples(x_train)
sx_train_zt, data_norm = normalize(sx_train)
params_train_zt, param_norm = normalize(params_train)

# train the posterior network
# define a network to approximate the posterior with a MoG
model_params_mdn = MultivariateMogMDN(ndim_input=2, ndim_output=2, n_hidden_units=10,
                                      n_hidden_layers=1, n_components=3)
optimizer = torch.optim.Adam(model_params_mdn.parameters(), lr=0.01)
trainer = Trainer(model_params_mdn, optimizer, verbose=True)

loss_trace = trainer.train(sx_train_zt, params_train_zt, n_epochs=200, n_minibatch=int(sx_train.shape[0] / 100))

result_dict = dict(prior_k=prior_k,
                   prior_theta=prior_theta,
                   sample_size=sample_size,
                   ntrain=ntrain,
                   model=model_nb,
                   xtrain=x_train,
                   sx_train=sx_train_zt,
                   data_norm=data_norm,
                   param_norm=param_norm,
                   mdn=model_params_mdn,
                   trainer=trainer)
full_path = '../data/learned_posterior_nbmodel_ntrain{}.p'.format(ntrain)

# save result
with open(full_path, 'wb') as outfile:
    pickle.dump(result_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)