import numpy as np
import scipy.stats

from multiprocessing import Pool


class BaseModel:
    def __init__(self, dim_param=1, sample_size=10, n_workers=1, seed=None):
        self.dim_param = dim_param
        self.sample_size = sample_size
        self.n_workers = n_workers

        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def gen(self, params_list):
        # generate data given parameter list

        # generate data given parameter list

        p = Pool(processes=self.n_workers)
        data_list = list(p.map(self.gen_single, params_list))

        return np.array(data_list)

    def gen_single(self, params):
        pass


class PoissonModel(BaseModel):
    def __init__(self, param_shape, param_scale, dim_param=1, sample_size=10, n_workers=1, seed=None):
        super().__init__(dim_param=dim_param, sample_size=sample_size, n_workers=n_workers, seed=seed)

        self.prior_lamb = scipy.stats.gamma(a=param_shape, scale=param_scale)

    # def gen(self, n_samples):
    #
    #     # set the seed
    #     np.random.seed(self.seed)
    #
    #     data_list = []
    #
    #     # generate a seeded list of random states
    #     random_states = np.random.randint(low=0, high=1000000, size=n_samples)
    #
    #     for sample_idx in range(n_samples):
    #         data_list.append(scipy.stats.poisson.rvs(mu=self.prior_lamb.rvs(random_state=random_states[sample_idx]),
    #                                                  size=self.sample_size,
    #                                                  random_state=random_states[sample_idx]))
    #
    #     return np.array(data_list)

    def gen_single(self, lam):

        return self.rng.poisson(lam=lam, size=self.sample_size)


class NegativeBinomialModel(BaseModel):
    def __init__(self, shape_prior_shape, shape_prior_scale,
                 scale_prior_shape, scale_prior_scale, dim_param=2, sample_size=10, n_workers=1, seed=None):
        super().__init__(dim_param=dim_param, sample_size=sample_size, n_workers=n_workers, seed=seed)

        self.prior_shape = scipy.stats.gamma(a=shape_prior_shape, scale=shape_prior_scale)
        self.prior_scale = scipy.stats.gamma(a=scale_prior_shape, scale=scale_prior_scale)

    # def gen(self, n_samples):
    #
    #     # set the seed
    #     np.random.seed(self.seed)
    #
    #     data_list = []
    #
    #     # generate a seeded list of random states
    #     random_states = np.random.randint(low=0, high=1000000, size=n_samples)
    #
    #     for sample_idx in range(n_samples):
    #
    #         # set random state
    #         rs = random_states[sample_idx]
    #         # for every sample, get a new gamma prior
    #         lambdas_from_gamma = scipy.stats.gamma.rvs(a=self.prior_shape.rvs(random_state=rs),
    #                                                    scale=self.prior_scale.rvs(random_state=rs),
    #                                                    size=self.sample_size, random_state=rs)
    #
    #         # now for every data point in the sample, to get NB, sample from that gamma prior into the poisson
    #         sample = []
    #         for ii in range(self.sample_size):
    #             # sample from poisson with lambdas sampled from gamma
    #             sample.append(scipy.stats.poisson.rvs(lambdas_from_gamma[ii]))
    #
    #         # add data set to samples
    #         data_list.append(sample)
    #
    #     return np.array(data_list)

    def gen_single(self, params):

        sample = []
        shape, scale = params

        # generate _samples from Gamma
        gamma_rv = scipy.stats.gamma(a=shape, scale=scale)

        for ii in range(self.sample_size):

            # sample from poisson with lambdas sampled from gamma
            sample.append(self.rng.poisson(lam=self.rng.gamma(shape=shape, scale=scale)))

        return sample
