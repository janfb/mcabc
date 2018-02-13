import numpy as np
import scipy.stats
import scipy.integrate
import time
from model_comparison.utils import *


def nb_evidence_integral(x, ks, thetas, integrant, log=False):

    k_grid, th_grid = np.meshgrid(ks, thetas)

    grid_values = np.zeros((thetas.size, ks.size))

    for i in range(thetas.shape[0]):
        for j in range(ks.shape[0]):
            grid_values[i, j] = integrant(k_grid[i, j], th_grid[i, j], x)

    integral = np.trapz(np.trapz(grid_values, x=thetas, axis=0), x=ks, axis=0)

    return np.log(integral) if log else integral


def nbinom_indirect_pmf(x, k, theta):
    gamma_pdf = scipy.stats.gamma(a=k, scale=theta)
    a = float(gamma_pdf.ppf(1e-8))
    b = float(gamma_pdf.ppf(1 - 1e-8))

    fun = lambda lam, k, theta, x: scipy.stats.poisson.pmf(x, mu=lam) * scipy.stats.gamma.pdf(lam, a=k, scale=theta)

    pmf_values = []
    for ix in x.squeeze():
        # integrate over all lambdas
        pmf_value, rr = scipy.integrate.quad(func=fun, a=a, b=b, args=(k, theta, ix), epsrel=1e-10)
        pmf_values.append(pmf_value)

    return pmf_values


def nb_evidence_integrant_indirect(k, theta, x):
    pk = prior_k.pdf(k)
    ptheta = prior_theta.pdf(theta)

    value = np.log(nbinom_indirect_pmf(x, k, theta)).sum() + np.log(pk) + np.log(ptheta)

    return np.exp(value)


def nb_evidence_integrant_direct(k, theta, x):
    r = k
    p = theta / (1 + theta)

    pk = prior_k.pdf(k)
    # pp = prior_theta.pdf(theta)
    pp = np.power(1 - p, -2) * prior_theta.pdf(theta)

    value = np.log(nbinom_pdf(x, r, p)).sum() + np.log(pk) + np.log(pp)

    return np.exp(value)


n_steps = 1000
sample_size = 2
n_samples = 1
seed = 1
time_stamp = time.strftime('%Y%m%d%H%M_')
figure_folder = '../figures/'

# set prior parameters
# set the shape or scale of the Gamma prior for the Poisson model
k1 = 9.0
# set the shape and scale of the prior on the shape of the Gamma for the mixture to be broad
theta2 = 2.0
k2 = 5.
# set the shape and scale of the prior on the scale of the Gamma for the mixture to be small
# this will make the variance and could be the tuning point of the amount of overdispersion / difficulty
theta3 = 1.0
k3 = 1

# then the scale of the Gamma prior for the Poisson is given by
theta1 = (k2 * theta2 * k3 * theta3) / k1

# get analytical means
mean_ana_poi = k1 * theta1
mean_ana_nb = k2 * k3 * theta2 * theta3

# set the priors
prior_k = scipy.stats.gamma(a=k2, scale=theta2)
prior_theta = scipy.stats.gamma(a=k3, scale=theta3)

# draw sample(s)
params_nb, X, lambs = sample_poisson_gamma_mixture(prior_k, prior_theta, n_samples, sample_size)

# set up a grid of values around the priors
# take grid over the whole range of the priors
ks = np.linspace(scipy.stats.gamma.ppf(1e-8, a=k2),
                 scipy.stats.gamma.ppf(1 - 1e-8, a=k2), n_steps)

thetas = np.linspace(scipy.stats.gamma.ppf(1e-8, a=k3),
                     scipy.stats.gamma.ppf(1 - 1e-8, a=k3), n_steps)

# ml1 = nb_evidence_integral(X, ks, thetas, integrant=nb_evidence_integrant_direct, log=False)
# print(ml1)
# k, theta = ks[10], thetas[10]
# p = theta / (1 + theta)
#
# print(nbinom_indirect_pmf(X.squeeze(), k, theta))
# print(nbinom_pdf(X.squeeze(), k, p))

ml2 = scipy.integrate.dblquad(func=nb_evidence_integrant_direct, a=thetas[0], b=thetas[-1],
                              gfun=lambda x: ks[0], hfun=lambda x: ks[-1], args=X)
print(ml2)

ml3 = scipy.integrate.dblquad(func=nb_evidence_integrant_indirect, a=thetas[0], b=thetas[-1],
                              gfun=lambda x: ks[0], hfun=lambda x: ks[-1], args=X)
print(ml3)