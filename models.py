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
    def __init__(self, dim_param=1, sample_size=10, n_workers=1, seed=None):
        super().__init__(dim_param=dim_param, sample_size=sample_size, n_workers=n_workers, seed=seed)

    def gen_single(self, lam):

        return self.rng.poisson(lam=lam, size=self.sample_size)


class NegativeBinomialModel(BaseModel):
    def __init__(self, dim_param=2, sample_size=10, n_workers=1, seed=None):
        super().__init__(dim_param=dim_param, sample_size=sample_size, n_workers=n_workers, seed=seed)

    def gen_single(self, params):

        sample = []
        shape, scale = params

        for ii in range(self.sample_size):

            # sample from poisson with lambdas sampled from gamma
            sample.append(self.rng.poisson(lam=self.rng.gamma(shape=shape, scale=scale)))

        return sample
