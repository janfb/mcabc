import numpy as np
import scipy.stats

from model_comparison.model.BaseModel import BaseModel
from model_comparison.utils.stats import NBExactPosterior


class NegativeBinomialModel(BaseModel):
    def __init__(self, dim_param=2, sample_size=10, n_workers=1, seed=None):
        super().__init__(dim_param=dim_param, sample_size=sample_size, n_workers=n_workers, seed=seed)

    def gen_single(self, params):
        # in multiprocessing the parameter vector additionally contains a seed
        if self.run_parallel:
            shape, scale, seed = params
            self.rng.seed(int(seed))
        else:
            shape, scale = params

        sample = []
        for ii in range(self.sample_size):

            # sample from poisson with lambdas sampled from gamma
            sample.append(self.rng.poisson(lam=self.rng.gamma(shape=shape, scale=scale)))

        return sample

    @staticmethod
    def get_exact_posterior(theta_obs, x_obs, prior_k, prior_theta, prec=1e-5, n_samples=200):
        """
        Get the exact posterior by numerical integration given the observed data and the priors.

        :param theta_obs: observed, true parameter
        :param x_obs: observed data, array of counts
        :param prior_k: scipy.stats.gamma object
        :param prior_theta: scipy.stats.gamma object
        :param prec: lower and upper tails in the coverage of the priors
        :param n_samples: number of sample points along each dimension of the grid

        :return: NBExactPosterior object with calculated posterior and samples
        """
        # set up the posterior
        post = NBExactPosterior(x_obs, prior_k, prior_theta)
        return post.calculat_exact_posterior(theta_obs, verbose=False, prec=prec, n_samples=n_samples)
