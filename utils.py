import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from random import shuffle
from scipy.stats import gamma, beta, nbinom, poisson
import scipy
from scipy.special import gammaln, betaln
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


def sample_poisson_gamma_mixture(prior1, prior2, n_samples, sample_size):
    thetas = []
    samples = []
    lambs = []

    for sample_idx in range(n_samples):

        # for every sample, get a new gamma prior
        thetas.append([prior1.rvs(), prior2.rvs()])
        gamma_prior = scipy.stats.gamma(a=thetas[sample_idx][0], scale=thetas[sample_idx][1])

        # now for every data point in the sample, to get NB, sample from that gamma prior into the poisson
        sample = []
        ls = []
        for ii in range(sample_size):
            ls.append(gamma_prior.rvs())
            sample.append(scipy.stats.poisson.rvs(ls[ii]))

        # add data set to samples
        samples.append(sample)
        lambs.append(ls)

    return np.array(thetas), np.array(samples), np.array(lambs)


def nbinom_pmf(k, r, p, axis=-1):
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
        pmf_value, rr = scipy.integrate.quad(func=fun,
                                             a=float(gamma_pdf.ppf(1e-8)),
                                             b=float(gamma_pdf.ppf(1 - 1e-8)),
                                             args=(k, theta, ix), epsrel=1e-10)
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


def nb_evidence_integrant_direct(k, theta, x, prior_k, prior_theta):
    """
    Negative Binomial marginal likelihood integrant: NB likelihood times prior pds values for given set of prior params
    """
    # set prior params for direct NB given params for indirect Poisson-Gamma mixture (Gamma priors on k and theta)
    r = k
    p = theta / (1 + theta)

    # get pdf values
    pk = prior_k.pdf(k)
    # do change of variables or not?
    pp = np.power(1 - p, -2) * prior_theta.pdf(theta)

    value = np.log(nbinom_pmf(x, r, p)).sum() + np.log(pk) + np.log(pp)

    return np.exp(value)


def calculate_pprob_from_evidences(pd1, pd2, priors=None): 
    if priors is None: 
        # p(m|d) = p(d | m) * p(m) / (sum_ p(d|m_i)p(m))))
        # because the prior is uniform we just return the normalized evidence: 
        return pd1 / (pd1 + pd2)
    else: 
        # p(m|d) = p(d | m) * p(m) / (sum_ p(d|m_i)p(m))))
        return pd1 * priors[0] / (pd1 * priors[0] + pd2 * priors[1])