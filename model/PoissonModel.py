import numpy as np
import scipy.stats

from model_comparison.model.BaseModel import BaseModel


class PoissonModel(BaseModel):
    def __init__(self, dim_param=1, sample_size=10, n_workers=1, seed=None):
        super().__init__(dim_param=dim_param, sample_size=sample_size, n_workers=n_workers, seed=seed)
        self.posterior = None

    def gen_single(self, params):
        # in multiprocessing the parameter vector additionally contains a seed
        if self.run_parallel:
            lam, seed = params
            self.rng.seed(int(seed))
        else:
            lam = params
        return self.rng.poisson(lam=lam, size=self.sample_size)

    def get_exact_posterior(self, x_obs, k, theta):
        """
        Given observed data and conjugate prior parameters, calculate the exact gamma posterior over lambda
        :param x_obs:
        :param k: prior shape
        :param theta: prior scale
        :return: scipy.stats.gamma object
        """

        sample_size = x_obs.size
        # get analytical gamma posterior
        k_post = k + np.sum(x_obs)

        # use the posterior given the summary stats, not the data vector
        scale_post = 1. / (sample_size + theta ** -1)

        # return gamma posterior
        self.posterior = scipy.stats.gamma(a=k_post, scale=scale_post)
        return self.posterior

    def get_mle_posterior(self, n_samples):
        """
        Sample from the exact posterior a lot and use the sample for a MLE estimate of the same posterior.

        Used for estimating a baseline for the DKL between the exact and the predicted posterior.
        :param n_samples:
        :return:
        """
        assert self.posterior is not None, 'you first have to calculate the exact posterior given observed data'

        # sample from exact posterior
        x = self.posterior.rvs(n_samples)

        # use mle estimate from wikipedia:
        s = np.log(x.mean()) - np.log(x).mean()
        k = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
        theta = np.sum(x) / (x.size * k)

        # return corresponding gamma posterior
        return scipy.stats.gamma(a=k, scale=theta)
