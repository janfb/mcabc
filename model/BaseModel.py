import numpy as np

from multiprocessing import Pool


class BaseModel:
    """
    Base class for a data generating model.
    """
    def __init__(self, dim_param=1, sample_size=10, n_workers=1, seed=None):
        """
        :param dim_param: number of parameters in the model
        :param sample_size: number of samples drawn in a single generation step
        :param n_workers: number of cores for multiprocessing
        :param seed:
        """
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
