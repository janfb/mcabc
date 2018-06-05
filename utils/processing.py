import numpy as np
import scipy

from scipy.stats import nbinom, poisson


def calculate_mse(fy, y):

    batch_se = np.power(fy - y, 2)

    mse = np.mean(batch_se)

    return mse


def generate_poisson_nb_data_set(n_samples, sample_size, prior_lam, prior_k, prior_theta,
                                 matched_means=False):

    lambs, x1 = sample_poisson(prior_lam, int(n_samples / 2), sample_size)
    if matched_means:
        thetas, x2 = sample_poisson_gamma_mixture_matched_means(prior_k, lambs, int(n_samples / 2), sample_size)
    else:
        thetas, x2 = sample_poisson_gamma_mixture(prior_k, prior_theta, int(n_samples / 2), sample_size)

    # join data
    x = np.vstack((x1, x2))

    # define model indices
    m = np.hstack((np.zeros(x1.shape[0]), np.ones(x2.shape[0]))).squeeze().astype(int).tolist()

    return x, m


def sample_poisson_gamma_mixture_matched_means(prior1, lambs, n_samples, sample_size):
    thetas = []
    samples = []

    for sample_idx in range(n_samples):
        # for every sample, get a new gamma prior
        r = prior1.rvs()
        theta = lambs[sample_idx] / r
        thetas.append([r, theta])
        gamma_prior = scipy.stats.gamma(a=thetas[sample_idx][0], scale=thetas[sample_idx][1])
        lambdas_from_gamma = scipy.stats.gamma.rvs(a=thetas[sample_idx][0], scale=thetas[sample_idx][1],
                                                   size=sample_size)

        # now for every data point in the sample, to get NB, sample from that gamma prior into the poisson
        sample = []
        for ii in range(sample_size):
            sample.append(scipy.stats.poisson.rvs(lambdas_from_gamma[ii]))

        # add data set to samples
        samples.append(sample)

    return np.array(thetas), np.array(samples)


def generate_poisson(N, prior):
    # sample from prior
    theta = prior.rvs()
    # generate samples
    x = poisson.rvs(mu=theta, size=N)

    return theta, x


def calculate_stats(x):
    # return [np.sum(x).astype(float), np.std(x).astype(float)]
    return np.array([np.mean(x).astype(float)])


def calculate_stats_toy_examples(x):

    if x.ndim == 1:
        x = x.reshape(1, x.shape[0])

    return np.vstack((np.mean(x, axis=1), np.var(x, axis=1))).T


def generate_negbin(N, r, prior):
    # sample from prior
    theta = prior.rvs()

    # generate samples
    x = nbinom.rvs(r, theta, size=N)

    return theta, x


def normalize(X, norm=None):
    if norm is None:
        xmean = X.mean(axis=0)
        xstd = X.std(axis=0)
    else:
        xmean = norm[0]
        xstd = norm[1]
    return (X - xmean) / xstd, (xmean, xstd)


def generate_nd_gaussian_dataset(n_samples, sample_size, prior, data_cov=None):

    X = []
    thetas = []
    ndims = prior.mean.size

    if data_cov is None:
        data_cov = np.eye(ndims)

    for i in range(n_samples):
        # sample from the prior
        theta = prior.rvs()

        # generate samples with mean from prior and unit variance
        x = scipy.stats.multivariate_normal.rvs(mean=theta, cov=data_cov, size=sample_size).reshape(sample_size, ndims)

        sx = np.array([np.sum(x, axis=0).astype(float)])

        # as data we append the summary stats
        X.append(sx)
        thetas.append([theta])

    return np.array(X).squeeze(), np.array(thetas).squeeze()


def sample_poisson(prior, n_samples, sample_size):
    thetas = []
    samples = []

    for sample_idx in range(n_samples):
        thetas.append(prior.rvs())
        samples.append(scipy.stats.poisson.rvs(mu=thetas[sample_idx], size=sample_size))

    return np.array(thetas), np.array(samples)


def sample_poisson_gamma_mixture(prior_k, prior_theta, n_samples, sample_size):
    """
    Generate samples from negative binomial distribution with specified priors.
    :param prior_k: scipy.stats.gamma object with parameters set. prior on Gamma shape
    :param prior_theta: scipy.stats.gamma object with parameters set. prior on Gamma scale
    :param n_samples: number of data sets sampled
    :param sample_size: number of samples per data set
    :return: parameters, data_sets
    """
    thetas = []
    samples = []

    for sample_idx in range(n_samples):

        # for every sample, get a new gamma prior
        thetas.append([prior_k.rvs(), prior_theta.rvs()])
        lambdas_from_gamma = scipy.stats.gamma.rvs(a=thetas[sample_idx][0], scale=thetas[sample_idx][1],
                                                   size=sample_size)

        # now for every data point in the sample, to get NB, sample from that gamma prior into the poisson
        sample = []
        for ii in range(sample_size):

            # sample from poisson with lambdas sampled from gamma
            sample.append(scipy.stats.poisson.rvs(lambdas_from_gamma[ii]))

        # add data set to samples
        samples.append(sample)

    return np.array(thetas), np.array(samples)