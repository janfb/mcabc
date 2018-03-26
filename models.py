import numpy as np
import scipy.stats

from multiprocessing import Pool
from model_comparison.utils import NBExactPosterior


class BaseModel:
    def __init__(self, dim_param=1, sample_size=10, n_workers=1, seed=None):
        self.dim_param = dim_param
        self.sample_size = sample_size
        self.n_workers = n_workers
        self.run_parallel = self.n_workers > 1

        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def gen(self, params_list):
        # generate data given parameter list

        params_list = np.array(params_list)
        n_params = params_list.shape[0]

        if self.run_parallel:
            # for multiprocessing the seeds for the individual processes have to
            # be prellocated.
            process_seeds = self.rng.randint(low=10000000, size=n_params)
            # add them to the parameter vector of every sample
            params_list_with_seeds = np.hstack((params_list.reshape(params_list.shape[0], -1),
                                                process_seeds.reshape(process_seeds.shape[0], -1))).tolist()
            # run in parallel
            p = Pool(processes=self.n_workers)
            data_list = list(p.map(self.gen_single, params_list_with_seeds))
        else:
            # if sequential, run as for loop over parameters
            data_list = []
            for param in params_list:
                data_list.append(self.gen_single(param))

        return np.array(data_list)

    def gen_single(self, params):
        pass


class PoissonModel(BaseModel):
    def __init__(self, dim_param=1, sample_size=10, n_workers=1, seed=None):
        super().__init__(dim_param=dim_param, sample_size=sample_size, n_workers=n_workers, seed=seed)

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
        return scipy.stats.gamma(a=k_post, scale=scale_post)


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

    def get_exact_posterior(self, x_obs, prior_k, prior_theta):
        """
        Get the exact posterior by numerical integration given the observed data and the priors
        :param x_obs: observed data, array of counts
        :param prior_k: scipy.stats.gamma object
        :param prior_theta: scipy.stats.gamma object
        :return: NBExactPosterior object with calculated posterior and samples
        """
        post = NBExactPosterior(x_obs, prior_k, prior_theta)
        # calculate posterior
        post.calculat_exact_posterior(verbose=False)
        # generate a lot of samples to approximate the mean
        samples = post.gen(10000)

        return post