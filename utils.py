import numpy as np
import torch
import torch.nn as nn
import tqdm

from torch.autograd import Variable
from random import shuffle
from scipy.stats import gamma, beta, nbinom, poisson
import scipy
import scipy.integrate
from scipy.special import gammaln, betaln, digamma
import matplotlib.pyplot as plt
import os


def log_betafun(a, b):
    # beta function is defined in terms of gamma as: B(a, b) = gamma(a)gamma(b)/gamma(a + b)
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


def beta_pdf(mu, a, b, log=True):
    result = (a - 1) * torch.log(mu) + (b - 1) * torch.log(1 - mu) - log_betafun(a, b)
    return result if log else torch.exp(result)


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


def batch_generator(dataset, batch_size=5):
    shuffle(dataset)
    N_full_batches = len(dataset) // batch_size
    for i in range(N_full_batches):
        idx_from = batch_size * i
        idx_to = batch_size * (i + 1)
        xs, ys = zip(*[(x, y) for x, y in dataset[idx_from:idx_to]])
        yield xs, ys


def poisson_evidence(x, k, theta, log=False):
    """
    Calculate evidence of a sample x under a Poisson model with Gamma prior.

    E(x) = gamma(k + sx) / ( gamma(k) * theta**k ) * (N + 1/theta)**(-k - sx) / prod x_i!

    Parameters
    ----------
    x : array-like
        batch of samples, shape (batch_size, )
    k : float
        shape parameter of the gamma prior
    theta: float
        scale parameter of the gamma prior
    log: bool
        if True, the log evidence is returned

    Returns
    -------
    log_evidence: float
        the log evidence (if log=True) or the evidence of the data
    """
    sx = np.sum(x)
    n_batch = x.size

    log_evidence = gammaln(k + sx) - (gammaln(k) + k * np.log(theta)) - (k + sx) * np.log(n_batch + theta ** -1) \
                   - np.sum(gammaln(np.array(x) + 1))

    return log_evidence if log else np.exp(log_evidence)


def poisson_sum_evidence(x, k, theta, log=True):
    """
    Calculate the evidence of the summary statistics of a sample under a Poisson model with Gamma prior.
    Given a batch of samples calculate the evidence (marginal likelihood) of the sufficient statistics
    (sum over the sample). Note the difference ot the poisson_evidence() method that calculates the evidence of the
    whole data sample.

    E(sx) = gamma(k + sx) / ( gamma(k) * (N*theta)**k ) * (1 + 1/(N*theta))**(-k - sx) / (sum x_i)!

    Parameters
    ----------
    x : array-like
        batch of samples, shape (batch_size, )
    k : float
        shape parameter of the gamma prior
    theta: float
        scale parameter of the gamma prior
    log: bool
        if True, the log evidence is returned

    Returns
    -------
    log_evidence: float
        the log evidence (if log=True) or the evidence of the data
    """

    n_batch = x.size
    sx = np.sum(x)

    result = -k * np.log(theta * n_batch) - gammaln(k) - gammaln(sx + 1) + gammaln(k + sx) - \
             (k + sx) * np.log(1 + (theta * n_batch)**-1)

    return result if log else np.exp(result)


def nbinom_evidence(x, r, a, b, log=False):
    """
    Calculate the evidence of a sample x under a negative binomial model with fixed r and beta prior on the success
    probability p.

    E(x) = \prod gamma(x_i +r) / ( gamma(x_i + 1) * gamma(r)**N ) * B(a + sx, b + Nr) / B(a, b)

    Parameters
    ----------
    x : array-like
        batch of samples, shape (batch_size, )
    r : int
        number of successes of the nbinom process
    a: float
        shape parameter alpha of the beta prior
    b: float
        shape parameter beta of the beta prior
    log: bool
        if True, the log evidence is returned

    Returns
    -------
    log_evidence: float
        the log evidence (if log=True) or the evidence of the data
    """
    b_batch = x.size
    sx = np.sum(x)

    fac = np.sum(gammaln(np.array(x) + r) - (gammaln(np.array(x) + 1) + gammaln([r])))
    log_evidence = fac + betaln(a + sx, b + b_batch * r) - betaln(a, b)

    return log_evidence if log else np.exp(log_evidence)


def nbinom_evidence_scipy(x, r, a, b, log=False):
    b_batch = x.size
    sx = np.sum(x)

    fac = np.sum(gammaln(np.array(x) + r) - (gammaln(np.array(x) + 1) + gammaln([r])))
    log_evidence = fac + betaln(a + b_batch * r, b + sx) - betaln(a, b)

    return log_evidence if log else np.exp(log_evidence)


def nbinom_sum_evidence(x, r, a, b, log=True):

    """
    Calculate the evidence of a the sufficient statistics sx of a sample x under a negative binomial model with fixed
    r and beta prior on the success probability p.

    E(sx) = binom(sx + Nr - 1, sx) * B(a + sx, b + Nr) / B(a, b)

    Parameters
    ----------
    x : array-like
        batch of samples, shape (batch_size, )
    r : int
        number of successes of the nbinom process
    a: float
        shape parameter alpha of the beta prior
    b: float
        shape parameter beta of the beta prior
    log: bool
        if True, the log evidence is returned

    Returns
    -------
    log_evidence: float
        the log evidence (if log=True) or the evidence of the data
    """

    N = x.size
    sx = np.sum(x)
    bc = scipy.special.binom(sx + N * r - 1, sx)

    log_evidence = np.log(bc) + betaln(a + sx, b + N * r) - betaln(a, b)

    return log_evidence if log else np.exp(log_evidence)


# magical gammaln fun from pyro
def log_gamma(xx):
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]
    magic1 = 1.000000000190015
    magic2 = 2.5066282746310005
    x = xx - 1.0
    t = x + 5.5
    t = t - (x + 0.5) * torch.log(t)
    ser = Variable(torch.ones(x.size()) * magic1)
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


def gamma_pdf(x, shape, scale, log=False):
    alpha = shape
    beta = 1. / scale

    ll_1 = -beta * x
    ll_2 = (alpha - 1.0) * torch.log(x)
    ll_3 = alpha * torch.log(beta)
    ll_4 = -log_gamma(alpha)
    result = torch.sum(ll_1 + ll_2 + ll_3 + ll_4, -1)
    if log:
        return result
    else:
        return torch.exp(result)


def get_posterior(model, X_o, thetas, norm):
    data = calculate_stats(X_o)
    data, norm = normalize(data, norm)

    stats_o = np.array(data.astype(float)).reshape(1, 1)

    X_var = Variable(torch.Tensor(stats_o))

    # predict shape and scale for given sufficient stats
    (out_shape, out_scale) = model(X_var)

    # evaluate gamma pdf with learned shape and scale parameters
    post = gamma.pdf(x=thetas, a=out_shape.data.numpy(), scale=out_scale.data.numpy())

    return post


def beta_mdn_loss(out_shape, out_scale, y):
    result = beta_pdf(y, out_shape, out_scale, log=True)
    result = torch.mean(result)  # mean over batch
    return -result


def gamma_mdn_loss(out_shape, out_scale, y):
    result = gamma_pdf(y, out_shape, out_scale, log=True)
    result = torch.mean(result)  # mean over batch
    return -result


def get_resp_mat(y, mus, sigmas, alphas):
    """
    Calculate the matrix of responsibility estimates for EM algorithm given a batch of MoG estimates.
    """

    n_data = mus.size()[0]
    n_components = mus.size()[1]

    numerator_mat = torch.zeros(n_data, n_components)

    denom = torch.zeros(n_data)

    # calculate responsibility values for every data point
    for k in range(n_components):
        nume = alphas[:, k] * univariate_normal_pdf(y, mus[:, k], sigmas[:, k], log=False)
        denom = torch.add(denom, nume)
        numerator_mat[:, k] = nume

    # add dimension to denominator vector
    denom = denom.view(n_data, 1)
    # expand it to match size with num matrix
    denom = denom.repeat(1, n_components)

    # take the fraction
    resp_mat = numerator_mat / denom

    return resp_mat


def my_log_sum_exp(x, axis=None):
    """
    Apply log-sum-exp with subtraction of the largest element to improve numerical stability.
    """
    (x_max, idx) = torch.max(x, dim=axis, keepdim=True)

    return torch.log(torch.sum(torch.exp(x - x_max), dim=axis, keepdim=True)) + x_max


def univariate_mog_pdf(y, mus, sigmas, alphas, log=False):
    """
    Calculate the density values of a batch of variates given the corresponding mus, sigmas, alphas.
    Use log-sum-exp trick to improve numerical stability.

    return the (log)-probabilities of all the entries in the batch. Type: (n_batch, 1)-Tensor
    """

    n_data = mus.size()[0]
    n_components = mus.size()[1]

    log_probs_mat = Variable(torch.zeros(n_data, n_components))

    # gather component log probs in matrix with components as columns, rows as data points
    for k in range(n_components):
        mu = mus[:, k].unsqueeze(1)
        sigma = sigmas[:, k].unsqueeze(1)
        lprobs = univariate_normal_pdf(y, mu, sigma, log=True)
        log_probs_mat[:, k] = lprobs

    log_probs_batch = my_log_sum_exp(torch.log(alphas) + log_probs_mat, axis=1)

    if log:
        result = log_probs_batch
    else:
        result = torch.exp(log_probs_batch)

    return result


def multivariate_mog_pdf(X, mus, Us, alphas, log=False):
    """
    Calculate pdf values for a batch of ND values under a mixture of multivariate Gaussians.

    Parameters
    ----------
    X : Pytorch Varibale containing a Tensor
        batch of samples, shape (batch_size, ndims)
    mus : Pytorch Varibale containing a Tensor
        means for every sample and mixture component, shape (batch_size, ndims, ncomponents)
    Us: Pytorch Varibale containing a Tensor
        upper triangle Choleski transform matrix of the precision matrices of every sample and component
        shape (batch_size, ncomponents, ndims, ndims)
    alphas: Pytorch Variable containing a Tensor
        mixture weights for every sample, shape (batch_size, ncomponents)
    log: bool
      if True, log probs are returned

    Returns
    -------
    result:  Variable containing a Tensor with shape (batch_size, 1)
        batch of density values, if log=True log probs
    """

    # get params
    n_batch, n_dims, n_components = mus.size()

    # prelocate matrix for log probs of each Gaussian component
    log_probs_mat = Variable(torch.zeros(n_batch, n_components))

    # take weighted sum over components to get log probs
    for k in range(n_components):
        log_probs_mat[:, k] = multivariate_normal_pdf(X=X, mus=mus[:, :, k], Us=Us[:, k, :, :], log=True)

    # now apply the log sum exp trick: sum_k alpha_k * N(Y|mu, sigma) = sum_k exp(log(alpha_k) + log(N(Y| mu, sigma)))
    # this give the log MoG density over the batch
    log_probs_batch = my_log_sum_exp(torch.log(alphas) + log_probs_mat, axis=1)  # sum over component axis=1

    # return log or linear density dependent on flag:
    if log:
        result = log_probs_batch
    else:
        result = torch.exp(log_probs_batch)

    return result


def gauss_pdf(y, mu, sigma, log=False):
    return univariate_normal_pdf(y, mu, sigma, log=log)


def univariate_normal_pdf(X, mus, sigmas, log=False):
    """
    Calculate pdf values for a batch of 1D Gaussian samples.

    Parameters
    ----------
    X : Pytorch Varibale containing a Tensor
        batch of samples, shape (batch_size, 1)
    mus : Pytorch Varibale containing a Tensor
        means for every sample, shape (batch_size, 1)
    sigmas: Pytorch Varibale containing a Tensor
        standard deviations for every sample, shape (batch_size, 1)
    log: bool
      if True, log probs are returned

    Returns
    -------
    result:  Variable containing a Tensor with shape (batch_size, 1)
        batch of density values, if log=True log probs
    """
    result = -0.5 * torch.log(2 * np.pi * sigmas ** 2) - 1 / (2 * sigmas ** 2) * (X.expand_as(mus) - mus) ** 2
    if log:
        return result
    else:
        return torch.exp(result)


def multivariate_normal_pdf(X, mus, Us, log=False):
    """
    Calculate pdf values for a batch of 2D Gaussian samples given mean and Choleski transform of the precision matrix.

    Parameters
    ----------
    X : Pytorch Varibale containing a Tensor
        batch of samples, shape (batch_size, ndims)
    mus : Pytorch Varibale containing a Tensor
        means for every sample, shape (batch_size, ndims)
    Us: Pytorch Varibale containing a Tensor
        Choleski transform of precision matrix for every sample, shape (batch_size, ndims, ndims)
    log: bool
      if True, log probs are returned

    Returns
    -------
    result:  Variable containing a Tensor with shape (batch_size, 1)
        batch of density values, if log=True log probs
    """

    # dimension of the Gaussian
    D = mus.size()[1]

    # get the precision matrices over batches using matrix multiplication: S^-1 = U'U
    Sin = torch.bmm(torch.transpose(Us, 1, 2), Us)

    # use torch.bmm to calculate probs over batch vectorized
    log_probs = - 0.5 * torch.sum((X - mus).unsqueeze(-1) * torch.bmm(Sin, (X - mus).unsqueeze(-1)), dim=1)
    # calculate normalization constant over batch extracting the diagonal of U manually
    norm_const = (torch.sum(torch.log(Us[:, np.arange(D), np.arange(D)]), -1) - (D / 2) * np.log(2 * np.pi)).unsqueeze(
        -1)

    result = norm_const + log_probs

    if log:
        return result
    else:
        return torch.exp(result)


def calculate_multivariate_normal_mu_posterior(X, sigma, N, mu_0, sigma_0):

    """
    Calculate the posterior over the mean parameter mu of a multivariate normal distribution given the data X,
    the covariance of the data generating distribution and mean and covariance of the Gaussian prior on mu.

    Parameters
    ----------
    X : numpy array (N, ndims)
        batch of samples, shape (batch_size, ndims)
    sigma:  numpy array (ndims, ndims)
        covariance underlying the data
    N : int
        batch size
    mu_0: numpy array (ndims, )
        prior mean
    sigma_0: numpy array (ndims, ndims)
        prior covariance matrix

    Returns
    -------
    mu_N: numpy array (ndims, )
        mean of the posterior
    sigma_N: numpy array (ndims, ndims)
        covariance matrix of the posterior
    """

    # formulas from Bishop p92/93
    sigma_N = np.linalg.inv(np.linalg.inv(sigma_0) + N * np.linalg.inv(sigma))
    mu_N = sigma_N.dot(N * np.linalg.inv(sigma).dot(X.mean(axis=0)) + np.linalg.inv(sigma_0).dot(mu_0))

    return mu_N, sigma_N


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


def save_figure(filename, time_stamp, folder='figures'):
    plt.savefig(os.path.join(folder, time_stamp + filename + '.png'), dpi=300)


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


def nbinom_pmf(k, r, p):
    """
    Calculate pmf values according to Wikipedia definition of the negative binomial distribution:
    p(X=x | r, p = (x + r - 1)choose(x) p^x (1 - p)^r
    """

    return scipy.special.binom(k + r - 1, k) * np.power(p, k) * np.power(1-p, r)


def nbinom_pmf_indirect(x, k, theta):
    """
    Calculate pmf values using the indirect sampling scheme via the Poisson-Gamma mixture. It is

        p(X=x | k, theta) = \int poisson(x | \lambda) gamma(\lambda | k, \theta) d\lambda

    The integral over \lambda is calculated using the double integration method from scipy.integrate.dbpquad
    """

    # set the gamma mixture pdf
    gamma_pdf = scipy.stats.gamma(a=k, scale=theta)

    # define integrant function: poisson(x | \lambda) gamma(\lambda | k, \theta)
    fun = lambda lam, ix: scipy.stats.poisson.pmf(ix, mu=lam) * scipy.stats.gamma.pdf(lam, a=k, scale=theta)

    # for every sample
    pmf_values = []
    for ix in x.squeeze():
        # integrate over all lambdas
        (pmf_value, rr) = scipy.integrate.quad(func=fun,
                                             a=float(gamma_pdf.ppf(1e-8)),
                                             b=float(gamma_pdf.ppf(1 - 1e-8)),
                                             args=[ix], epsrel=1e-10)
        pmf_values.append(pmf_value)

    return pmf_values


def nb_evidence_grid_integral(x, prior_k, prior_theta, ks, thetas, integrant, log=False):

    k_grid, th_grid = np.meshgrid(ks, thetas)

    grid_values = np.zeros((thetas.size, ks.size))

    for i in range(thetas.shape[0]):
        for j in range(ks.shape[0]):
            grid_values[i, j] = integrant(k_grid[i, j], th_grid[i, j], x, prior_k, prior_theta)

    integral = np.trapz(np.trapz(grid_values, x=thetas, axis=0), x=ks, axis=0)

    return np.log(integral) if log else integral


def nb_evidence_integrant_indirect(k, theta, x, prior_k, prior_theta):
    """
    Negative Binomial marginal likelihood integrant for the indirect sampling method via Poisson-Gamma mixture.
    """

    # get the prior pdf values for k and theta
    pk = prior_k.pdf(k)
    ptheta = prior_theta.pdf(theta)

    # evaluate the pmf and take the product (log sum) over samples, multiply with prior pds values
    value = np.log(nbinom_pmf_indirect(x, k, theta)).sum() + np.log(pk) + np.log(ptheta)

    # return exponential
    return np.exp(value)


def nb_evidence_integrant_direct(r, p, x, prior_k, prior_theta):
    """
    Negative Binomial marginal likelihood integrant: NB likelihood times prior pds values for given set of prior params
    """
    # set prior params for direct NB given params for indirect Poisson-Gamma mixture (Gamma priors on k and theta)

    # get pdf values
    pr = prior_k.pdf(r)
    # do change of variables or not?
    pp = np.power(1 - p, -2) * prior_theta.pdf(p / (1 - p))

    value = np.log(nbinom_pmf(x, r, p)).sum() + np.log(pr) + np.log(pp)

    return np.exp(value)


def calculate_pprob_from_evidences(pd1, pd2, priors=None):
    if priors is None:
        # p(m|d) = p(d | m) * p(m) / (sum_ p(d|m_i)p(m))))
        # because the prior is uniform we just return the normalized evidence:
        return pd1 / (pd1 + pd2)
    else:
        # p(m|d) = p(d | m) * p(m) / (sum_ p(d|m_i)p(m))))
        return pd1 * priors[0] / (pd1 * priors[0] + pd2 * priors[1])


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


def calculate_nb_evidence(x, k_k, theta_k, k_theta, theta_theta, log=False):
    # set up a grid of values around the priors
    # take grid over the whole range of the priors

    k_start = scipy.stats.gamma.ppf(1e-8, a=k_k)
    k_end = scipy.stats.gamma.ppf(1 - 1e-8, a=k_k)

    theta_start = scipy.stats.gamma.ppf(1e-8, a=k_theta)
    theta_end = scipy.stats.gamma.ppf(1 - 1e-8, a=k_theta)

    # set priors
    prior_k = scipy.stats.gamma(a=k_k, scale=theta_k)
    prior_theta = scipy.stats.gamma(a=k_theta, scale=theta_theta)

    (evidence, err) = scipy.integrate.dblquad(func=nb_evidence_integrant_direct,
                                              a=theta_start / (1 + theta_start),
                                              b=theta_end / (1 + theta_end),
                                              gfun=lambda x: k_start, hfun=lambda x: k_end,
                                              args=[x, prior_k, prior_theta])

    return np.log(evidence) if log else evidence


def calculate_mse(fy, y):

    batch_se = np.power(fy - y, 2)

    mse = np.mean(batch_se)

    return mse


def get_mog_posterior(model, stats_o, thetas):
    (out_alpha, out_sigma, out_mu) = model(Variable(torch.Tensor(stats_o)))

    n_thetas = thetas.size
    n_batch, n_components = out_mu.size()

    # get parameters in torch format
    torch_thetas = Variable(torch.Tensor(thetas)).unsqueeze(1)
    sigmas = out_sigma.expand(n_thetas, n_components)
    mus = out_mu.expand(n_thetas, n_components)
    alphas = out_alpha.expand(n_thetas, n_components)

    # get predicted posterior as MoG
    post = univariate_mog_pdf(y=torch_thetas, sigmas=sigmas, mus=mus, alphas=alphas)

    return post


def calculate_gamma_dkl(k1, theta1, k2, theta2):
    return (k1 - k2) * digamma(k1) - gammaln(k1) + gammaln(k2) + \
           k2 * (np.log(theta2) - np.log(theta1)) + k1 * (theta1 - theta2) / theta2


def calculate_dkl_1D_scipy(p_pdf_array, q_pdf_array):
    """
    Calculate DKL from array of pdf values.

    The arrays should cover as much of the range as possible.
    :param p_pdf_array:
    :param q_pdf_array:
    :return:
    """
    return scipy.stats.entropy(pk=p_pdf_array, qk=q_pdf_array)


def calculate_dkl_monte_carlo(x, p_pdf, q_pdf):
    """
    Estimate the DKL between 1D RV p and q.

    :param x: samples from p
    :param p_pdf: pdf function for p
    :param q_pdf: pdf function for q
    :return: estimate of dkl, standard error
    """

    # eval those under p and q
    pp = p_pdf(x)
    pq = q_pdf(x)

    # estimate expectation of log
    log = np.log(pp) - np.log(pq)
    dkl = log.mean()
    # estimate the standard error
    stderr = log.std(ddof=1) / np.sqrt(x.shape[0])

    return dkl, stderr


def calculate_dkl(p, q):
    """
    Calculate dkl between p and q.
    :param p: scipy stats object with .pdf() and .ppf() methods
    :param q: delfi.distribution object with .eval() method
    :return: dkl(p, q)
    """
    # parameter range
    p_start = p.ppf(1e-9)
    p_end = p.ppf(1 - 1e-9)

    # integral function
    def integrant(x):
        log_pp = p.logpdf(x)
        log_pq = q.eval(np.reshape(x, (1, -1)), log=True)
        return np.exp(log_pp) * (log_pp - log_pq)

    (dkl, err) = scipy.integrate.quad(integrant, a=p_start, b=p_end)
    return dkl


def calculate_credible_intervals_success(theta, ppf_fun, intervals, args=None):
    """
    Calculate credible intervals given a true parameter value and a percent point function of a distribution
    :param theta: true parameter
    :param ppf_fun: percent point function (inverse CDF)
    :param intervals: array-like, credible intervals to be calculated
    :param args: arguments to the ppf function
    :return: a binary vector, same length as intervals, indicating whether the true parameter lies in that interval
    """
    tails = (1 - intervals) / 2

    # get the boundaries of the credible intervals
    lows, highs = ppf_fun(tails, *args), ppf_fun(1 - tails, *args)
    success = np.ones_like(intervals) * np.logical_and(lows <= theta, theta <= highs)

    return success


def check_credible_regions(theta_o, cdf_fun, credible_regions):

    q = cdf_fun(theta_o)

    if q > 0.5:
        # the mass in the CR is 1 - how much mass is above times 2
        cr_mass = 1 - 2 * (1 - q)
    else:
        # or 1 - how much mass is below, times 2
        cr_mass = 1 - 2 * q
    counts = np.ones_like(credible_regions) * (credible_regions > cr_mass)
    return counts


def calculate_ppf_from_samples(qs, samples):
    """
    Given quantiles and samples, calculate values corresponding to the quantiles by approximating the
    MoG inverse CDF from samples.
    :param qs: quantiles, array-like
    :param samples: number of samples used to for sampling
    :return: corresponding values, array-like
    """

    qs = np.atleast_1d(qs)
    values = np.zeros_like(qs)

    # use bins from min to max
    bins = np.linspace(samples.min(), samples.max(), 1000)
    # asign samples to bins
    bin_idx = np.digitize(samples, bins)
    # count samples per bin --> histogram
    n = np.bincount(bin_idx.squeeze())
    # take the normalized cum sum as the cdf
    cdf = np.cumsum(n) / np.sum(n)

    # for every quantile, get the corresponding value on the cdf
    for i, qi in enumerate(qs):
        quantile_idx = np.where(cdf >= qi)[0][0]
        values[i] = bins[quantile_idx]

    return values


def inverse_transform_sampling_1d(array, pdf_array, n_samples):
    """
    Generate samples from an arbitrary 1D distribution given an array of pdf values. Using inverse transform sampling.

    Calculates CDF by summing up values in the PDF. Assumes values in array and PDF are spaced uniformly.

    :param array: array of RV values covering a representative range
    :param pdf_array: the corresponding PDF values of the values in 'array'.
    :param n_samples: number of samples to generate
    :return: array-like, array of pseudo-randomly generated sampled.
    """
    uniform_samples = scipy.stats.uniform.rvs(size=n_samples)
    samples = np.zeros(n_samples)
    # calculate the cdf by taking the cumsum and normaliying by dt
    cdf = np.cumsum(pdf_array) * (array[1] - array[0])

    for i, s in enumerate(uniform_samples):
        # find idx in cmf
        idx = np.where(cdf >= s)[0][0]
        # add the corresponding value
        samples[i] = array[idx]

    return samples


def inverse_transform_sampling_2d(x1, x2, joint_pdf, n_samples):
    """
    Generate samples from an arbitrary 2D distribution f(x, y) given a matrix of joint density values.

    Using 2D inverse transform sampling: Calculate the marginal p(x1) and the condition p(x2 | x1). Generate
    pseudo random samples from the x1 marginal. Then generate pseudo-random samples from the conditional, each sample
    conditioned on a x1 sample of the previos step.
    :param x1: values of RV x1
    :param x2: values of RV x2
    :param joint_pdf: 2D array of PDF values corresponding to the bins defined in x1 and x2
    :param n_samples: number of samples to draw
    :return: np array with samples (n_samples, 2)
    """

    # calculate marginal of x1 by integrating over x2
    x1_pdf = np.trapz(joint_pdf, x=x2, axis=1)

    # sample from marginal
    samples_x1 = inverse_transform_sampling_1d(x1, x1_pdf, n_samples)

    # calculate the conditional of x2 given x1 using Bayes rule
    # this gives a matrix of pdf, one for each values of x1 that we condition on.
    x2_pdf = np.zeros_like(joint_pdf)
    x2_cdf = np.zeros_like(joint_pdf)
    # condition on every x1
    for i in range(x1.size):
        # conditioned on this x1, apply Bayes
        px1 = x1_pdf[i] if x1_pdf[i] > 0. else 1e-12
        x2_pdf[i, ] = joint_pdf[i, :] / px1
        # get the corresponding cdf by cumsum and normalization
        x2_cdf[i, ] = np.cumsum(x2_pdf[i,])
        x2_cdf[i, ] /= np.max(x2_cdf[i,])
        assert np.isclose(x2_cdf[i, 0], 0, atol=1e-5), 'cdf should go from 0 to 1, {}'.format(x2_cdf[i, 0])
        assert np.isclose(x2_cdf[i, -1], 1, atol=1e-5), 'cdf should go from 0 to 1, {}'.format(x2_cdf[i, 0])

    # sample new uniform numbers
    uniform_samples = scipy.stats.uniform.rvs(size=n_samples)

    samples_x2 = []
    for uni_sample, x1_sample in zip(uniform_samples, samples_x1):
        # get the index of the x1 sample for conditioning
        idx_x1 = np.where(x1 >= x1_sample)[0][0]
        # find idx in conditional cmf
        idx_u = np.where(x2_cdf[idx_x1,] >= uni_sample)[0][0]

        # add the corresponding value
        samples_x2.append(x2[idx_u])

    return np.vstack((samples_x1, np.array(samples_x2))).T


class NBExactPosterior:
    """
    Class for the exact NB posterior. Defined by observed data and priors on k and theta, the shape and scale of the
    Gamma distribution in the Poisson-Gamma mixture.

    Has methods to calculate the exact posterior in terms of a joint pdf matrix using numerical integration.
    And methods to evaluate and to generate samples under this pdf.

    Once the posterior is calculated and samples are generated, it has properties mean and std to be compared to the
    predicted posterior.
    """

    def __init__(self, x, prior_k, prior_theta):
        """
        Instantiate the posterior with data and priors. the actual posterior has to be calculate using
        calculate_exact_posterior()
        :param x: observed data, array of counts
        :param prior_k: scipy.stats.gamma object
        :param prior_theta: scipy.stats.gamma object
        """

        # set flags
        self.samples_generated = False  # whether mean and std are defined
        self.calculated = False  # whether exact solution has been calculated

        self.xo = x
        self.prior_k = prior_k
        self.prior_th = prior_theta

        # prelocate
        self.evidence = None
        self.joint_pdf = None
        self.joint_cdf = None
        self.ks = None
        self.thetas = None

        self.samples = []

    def calculat_exact_posterior(self, theta_o, n_samples=200, prec=1e-6, verbose=True):
        """
        Calculate the exact posterior.
        :param theta_o: the true parameter theta
        :param n_samples: the number of entries per dimension on the joint_pdf grid
        :param prec: precision for the range of prior values
        :return: No return
        """

        # if not calculated
        if not self.calculated:
            self.calculated = True
            # set up a grid. take into account the true theta value to cover the region around it in the posterior
            # get the quantiles of the true theto under the prior
            k_pos = self.prior_k.cdf(theta_o[0])
            th_pos = self.prior_th.cdf(theta_o[1])

            # set the tail around it,
            tail = 0.8
            # choose ranges such that there are enough left and right of the true theta, use prec for bounds
            self.ks = np.linspace(self.prior_k.ppf(np.max((prec, k_pos - tail))),
                                  self.prior_k.ppf(np.min((1 - prec, k_pos + tail))), n_samples)
            self.thetas = np.linspace(self.prior_th.ppf(np.max((prec, th_pos - tail))),
                                      self.prior_th.ppf(np.min((1 - prec, th_pos + tail))), n_samples)

            joint_pdf = np.zeros((self.ks.size, self.thetas.size))

            # calculate likelihodd times prior for every grid value
            with tqdm.tqdm(total=self.ks.size * self.thetas.size, desc='calculating posterior',
                           disable=not verbose) as pbar:

                for i, k in enumerate(self.ks):
                    for j, th in enumerate(self.thetas):
                        r = k
                        p = th / (1 + th)
                        joint_pdf[i, j] = nb_evidence_integrant_direct(r, p, self.xo, self.prior_k, self.prior_th)
                        pbar.update()

            # calculate the evidence as the integral over the grid of likelihood * prior values
            self.evidence = np.trapz(np.trapz(joint_pdf, x=self.thetas, axis=1), x=self.ks, axis=0)
            self.joint_pdf = joint_pdf / self.evidence

            # calculate cdf
            # Calculate CDF by taking cumsum on each axis
            s1 = np.cumsum(np.cumsum(self.joint_pdf, axis=0), axis=1)
            # approximate cdf by summation and normalization
            self.joint_cdf = s1 / s1.max()
        else:
            print('already done')

    def eval(self, x, log=False):
        """
        Evaluate the joint pdf for value pairs given in x.
        :param x: np.array, shape (n, 2)
        :return: pdf values, np array, shape (n, )
        """

        x = np.atleast_1d(x)
        assert self.calculated, 'calculate the joint posterior first using calculate_exaxt_posterior'
        assert x.ndim == 2, 'x should have two dimensions, (n_samples, 2)'
        assert x.shape[1] == 2, 'each datum should have two entries, [k, theta]'

        pdf_values = []
        # for each pair of (k, theta)
        for xi in x:
            # look up indices in the ranges
            idx_k = np.where(self.ks >= xi[0])[0][0]
            idx_th = np.where(self.thetas >= xi[1])[0][0]

            # take corresponding pdf values from pdf grid
            pdf_values.append(self.joint_pdf[idx_k, idx_th])

        return np.log(np.array(pdf_values)) if log else np.array(pdf_values)

    # to mimic scipy.stats behavior
    def pdf(self, x):
        """
        Evaluate pdf at x
        :param x: samples
        :return: density values
        """
        return self.eval(x)

    def logpdf(self, x):
        """
        Evaluate log density at x
        :param x: samples
        :return: log density
        """
        return self.eval(x, log=True)

    def ppf(self, q):
        """
        Percent point function at q, or inverse CDF. Approximated by looking up the index in the cdf table
        that is closest to q.
        :param q: quantile
        :return: corresponding value on the RV range
        """
        q = np.atleast_1d(q)

        # look up the index of the quantile in the 2D CDF grid
        values = []
        for qi in q:
            # find index in grid for every dimension
            idx1, idx2 = np.where(self.joint_cdf >= qi)
            values.append([self.ks[idx1[0]], self.thetas[idx2[0]]])

        return np.array(values)

    def cdf(self, x):

        x = np.atleast_1d(x)
        qs = []

        for xi in x:
            # find idx of x on the cdf grid
            idx_k = np.where(self.ks >= xi[0])[0][0]
            idx_th = np.where(self.thetas >= xi[1])[0][0]

            # get value from cdf
            qs.append(self.joint_cdf[idx_k, idx_th])

        return np.array(qs)

    def gen(self, n_samples):
        """
        Generate samples under the joint pdf grid using inverse transform sampling
        :param n_samples:
        :return:
        """

        assert self.calculated, 'calculate the joint posterior first using calculate_exaxt_posterior'
        self.samples_generated = True

        # generate new samples
        samples = inverse_transform_sampling_2d(self.ks, self.thetas, self.joint_pdf, n_samples)

        # add to list of all samples
        self.samples += samples.tolist()

        return samples

    def rvs(self, n_samples):
        return self.gen(n_samples)

    @property
    def mean(self):
        if len(self.samples) == 0:
            self.gen(1000)
        return np.mean(self.samples, axis=0).reshape(-1)

    @property
    def std(self):
        if len(self.samples) == 0:
            self.gen(1000)
        return np.sqrt(np.diag(np.cov(np.array(self.samples).T))).reshape(-1)

    @property
    def cov(self):
        if len(self.samples) == 0:
            self.gen(1000)
        return np.cov(np.array(self.samples).T)

    def get_marginals(self):

        k_pdf = np.trapz(self.joint_pdf, x=self.thetas, axis=1)
        th_pdf = np.trapz(self.joint_pdf, x=self.ks, axis=0)

        return [Distribution(self.ks, k_pdf), Distribution(self.thetas, th_pdf)]


class Distribution:
    """
    Class for arbitrary distribution defined in terms of an array of pdf values. Used for representing the marginals
    of the numerically calculated NB posterior.
    """

    def __init__(self, support_array, pdf_array):

        self.support = support_array
        self.pdf_array = pdf_array

        self.cdf_array = np.cumsum(self.pdf_array)
        self.cdf_array /= self.cdf_array.max()
        self.samples = []

    def eval(self, x, log=False):

        pdf_values = []
        # for each sample
        for xi in x:
            # look up index in the supported range
            idx_i = np.where(self.support >= xi)[0][0]

            # take corresponding pdf value from pdf
            pdf_values.append(self.pdf_array[idx_i])

        return np.log(np.array(pdf_values)) if log else np.array(pdf_values)

    def pdf(self, x):
        return self.eval(x)

    def logpdf(self, x):
        return self.eval(x, log=True)

    def gen(self, n_samples):
        """
        Generate samples under the pdf using inverse transform sampling
        :param n_samples:
        :return: array-like, samples
        """
        # generate samples
        samples = inverse_transform_sampling_1d(self.support, self.pdf_array, n_samples=n_samples)
        # add to all samples
        self.samples += samples.tolist()

        return samples

    def ppf(self, qs):
        """
        Percent point function at q, or inverse CDF. Approximated by looking up the index in the cdf table
        that is closest to q.
        :param q: quantile
        :return: corresponding value on the RV range
        """
        q = np.atleast_1d(qs)

        # look up the index of the quantile in the 2D CDF grid
        values = []
        for q in qs:
            # find index in grid for every dimension
            idx1 = np.where(self.cdf_array >= q)[0][0]
            values.append(self.support[idx1])

        return np.array(values)

    def cdf(self, xs):
        """
        Evaluate CDF at every x in xs. Approximated by looking up the index in the cdf array.
        :param xs: RV values to evaluate
        :return: quantiles in [0, 1]
        """
        # make it an array in case it is a scalar.
        xs = np.atleast_1d(xs)

        cdf_values = []
        for xi in xs:
            # look up index in the support array
            idx = np.where(self.support >= xi)[0][0]
            # get the corresponding quantile
            cdf_values.append(self.cdf_array[idx])

        return np.array(cdf_values)

    def get_credible_interval_counts(self, th, credible_intervals):
        # get the quantile of theta

        q = self.cdf(th)

        # q mass lies below th, therefore the CI is
        if q > 0.5:
            # for q > .5, 1 - how much mass is above q times 2 (2 tails)
            ci = 1 - 2 * (1 - q)
        else:
            # how much mass is below, times 2 (2 tails)
            ci = 1 - 2 * q
        counts = np.ones_like(credible_intervals) * (credible_intervals>= ci)
        return counts

    @property
    def mean(self):
        """
        Mean estimated from samples
        :return:
        """
        if len(self.samples) == 0:
            self.gen(1000)

        return np.mean(self.samples)

    @property
    def std(self):
        """
        Mean estimated from samples
        :return:
        """
        if len(self.samples) == 0:
            self.gen(1000)

        return np.std(self.samples)


class JointGammaPrior:

    def __init__(self, prior_k, prior_theta):

        self.prior_k = prior_k
        self.prior_theta = prior_theta

    def gen(self, n_samples):

        sk = self.prior_k.rvs(n_samples)
        sth = self.prior_theta.rvs(n_samples)

        return np.vstack((sk, sth)).reshape(n_samples, 2)

    def pdf(self, samples):

        samples = np.atleast_1d(samples)
        assert samples.shape[1] == 2, 'samples should be (n_samples, 2)'

        pk = self.prior_k.pdf(samples[:, 0])
        pth = self.prior_theta.pdf(samples[:, 1])

        return pk * pth

    def rvs(self, n_samples):
        return self.gen(n_samples)

    def eval(self, samples):
        return self.pdf(samples)


def rejection_sampling_abc(sxo, models, model_priors, param_priors, niter=10000, verbose=False, eps=1e-1):
    """
    Basic rejection sampling algorithm including estimate of the model posterior.

    Takes mean and var of data vector as summary stats.

    :param sxo: observed summary stats
    :param models: list with scipy.stats objects
    :param model_priors: list of model prior probs, summing to 1
    :param param_priors: list of lists containing the scipy.stats prior objects for every model
    :param niter: number of iterations
    :param verbose: no pbar if False
    :param eps: rejection criterion.
    :return: list of accepted model indices, list of lists of accepted parameters, list of differences for every it.
    """
    n_models = len(models)

    accepted_its = []
    accepted_mi = []
    accepted_params = [[] for n in range(n_models)]
    differences = []

    for it in tqdm.tqdm(range(niter), disable=not verbose):
        # sample model index
        mi = np.where(np.random.multinomial(n=1, pvals=model_priors))[0][0]

        # get model and prior
        m = models[mi]
        priors = param_priors[mi]
        # sample corresponding params
        params = [prior.rvs() for prior in priors]
        # get data and stats
        x = m.gen([params])
        sx = np.array([x.mean(), x.var()])
        # take diff
        d = np.abs(sxo - sx).sum()
        differences.append(d)
        if d < eps:
            accepted_its.append(it)
            accepted_params[mi].append(params)
            accepted_mi.append(mi)

    return accepted_mi, accepted_params, differences


def rejection_abc_from_stats(sxo, model_stats, model_priors, niter=10000, verbose=False, eps=1e1):
    """
    Rejection abc to estimate model posterior probability.

    Works with a precomputed set of summary statistics. Samples from this data set to approx. sampling form the
    distribution and running the forward simulations. The provided set of summary statistics should therefore be
    reasonably large.

    :param sxo: observed summary stats
    :param model_stats: list of sets of summary stats for every model
    :param model_priors: list of model prior probs, sums to 1
    :param niter: number of iterations
    :param verbose: shows pbar if True
    :param eps: rejection criterion, select carefully!
    :return:
        list: accepted model indices over iterations,
        list: lists with indices of accepted data sets, for parameter posterior estimation
        list: difference to observed stats for every iteration,
    """

    n_models = len(model_stats)

    accepted_its = []
    accepted_mi = []
    differences = []
    data_set_indices = [[] for n in range(n_models)]

    for it in tqdm.tqdm(range(niter), disable=not verbose):
        # sample model index
        mi = np.where(np.random.multinomial(n=1, pvals=model_priors))[0][0]

        # get data and stats
        data_set = model_stats[mi]
        idx = np.random.randint(0, data_set.shape[0])
        sx = data_set[idx,]
        # take diff
        d = np.abs(sxo - sx).sum()
        differences.append(d)
        if d < eps:
            # save idx of data used in this iteration
            data_set_indices[mi].append(idx)
            # save iteration
            accepted_its.append(it)
            # save model idx
            accepted_mi.append(mi)

    return accepted_mi, data_set_indices, differences
