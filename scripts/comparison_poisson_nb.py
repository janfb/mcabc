import numpy as np
import os
import scipy
import sys
import torch
import torch.nn as nn
import time

from random import shuffle
from scipy.stats import gamma, beta, nbinom, poisson
from scipy.special import gammaln, betaln
from torch.autograd import Variable

sys.path.append('../')
from model_comparison.utils import *
from model_comparison.mdns import ClassificationSingleLayerMDN, Trainer


# set params
sample_size = 10
n_samples = 10000
n_epochs = 100
n_minibatch = 100
n_test_samples = 20

# set RNG
seed = 2
np.random.seed(seed)

time_stamp = time.strftime('%Y%m%d%H%M_')

# set priors
k2 = 2.
theta2 = 5.0

k3 = 1
theta3 = .1

# then the scale of the Gamma prior for the Poisson is given by
k1 = 2.0
theta1 = (k2 * theta2 * k3 * theta3) / k1

prior_lam = scipy.stats.gamma(a=k1, scale=theta1)
prior_k = scipy.stats.gamma(a=k2, scale=theta2)
prior_theta = scipy.stats.gamma(a=k3, scale=theta3)
prior_r = prior_k
# define prior p by change of variables
def prior_p(p): np.power(1-p, -2) * prior_theta(p / (1 - p))


# generate the training data
X, m = generate_poisson_nb_data_set(n_samples, sample_size, prior_lam, prior_k, prior_theta, seed=seed)
SX = calculate_stats_toy_examples(X)
SX, training_norm = normalize(SX)

# define the nn and trainer
model = ClassificationSingleLayerMDN(ndim_input=2, n_hidden=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = Trainer(model, optimizer, verbose=True, classification=True)

# train with training data
loss_trace = trainer.train(SX, m, n_epochs=n_epochs, n_minibatch=n_minibatch)
plt.plot(loss_trace)
plt.show()

# generate test data
Xtest, mtest = generate_poisson_nb_data_set(n_test_samples, sample_size, prior_lam, prior_k, prior_theta, seed=seed)
SXtest = calculate_stats_toy_examples(Xtest)
SXtest, training_norm = normalize(SXtest, training_norm)

# predict test data
poisson_posterior_probs = []
softmax = nn.Softmax(dim=0)

for xtest in SXtest:
    print(xtest)
    # get activation
    out_act = model(Variable(torch.Tensor(xtest)))
    # normalize
    p_vec = softmax(out_act).data.numpy()
    # in this vector, index 0 is Poi, index 1 is NB
    ppoi = p_vec[0]
    poisson_posterior_probs.append(ppoi)

poisson_posterior_probs = np.array(poisson_posterior_probs)
# avoid overflow in log
poisson_posterior_probs[poisson_posterior_probs == 1.] = 0.99999
# calculate log bayes factor: equivalent to log ratio of posterior probs
logbf_predicted = np.log(poisson_posterior_probs) - np.log(1 - poisson_posterior_probs)

plt.plot(poisson_posterior_probs)
plt.show()

# calculate analytical log bayes factors
nb_logevidences = []
poi_logevidences = []

k_start = scipy.stats.gamma.ppf(1e-8, a=k2)
k_end = scipy.stats.gamma.ppf(1 - 1e-8, a=k2)

theta_start = scipy.stats.gamma.ppf(1e-8, a=k3)
theta_end = scipy.stats.gamma.ppf(1 - 1e-8, a=k3)

for ii, (x, mi) in enumerate(zip(Xtest, mtest)):

    (nb_logevi, err) = scipy.integrate.dblquad(func=nb_evidence_integrant_direct,
                                           a=theta_start / (1 + theta_start),
                                           b=theta_end / (1 + theta_end),
                                           gfun=lambda x: k_start, hfun=lambda x: k_end,
                                           args=[x, prior_k, prior_theta])

    #nb_logevi = calculate_nb_evidence(x, k2, theta2, k3, theta3, log=True)
    poi_logevi = le = poisson_evidence(x, k=k1, theta=theta1, log=True)

    nb_logevidences.append(nb_logevi)
    poi_logevidences.append(poi_logevi)

# calculate posterior probs from log evidences
poi_logevidences = np.array(poi_logevidences)
nb_logevidences = np.array(nb_logevidences)

# just subtract to get the ratio of evidences
logbf_ana = poi_logevidences - nb_logevidences

# the analytical poisson posterior is just the poisson evidence normalized with the sum of all evidences
ppoi_ana = poi_logevidences - (nb_logevidences + poi_logevidences)

# make some plots for testing
plt.plot(logbf_predicted, 'o-', label='predicted')
plt.plot(logbf_ana, 'o-', label='analytical')
plt.xlabel('test data set index')
plt.ylabel('log bayes factor')
plt.legend()
plt.tight_layout()
plt.show()

# set up the result dict

