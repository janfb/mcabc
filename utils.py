import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from random import shuffle
from scipy.stats import gamma, beta, nbinom, poisson
import scipy
from scipy.special import gammaln, betaln


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
    return [np.sum(x).astype(float)]


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


def poisson_evidence(x, k, theta, N, log=False):
    """
    k : shape parameter for gamma
    theta : scale parameter for gamma
    """
    x_sum = np.sum(x)
    log_xfac = np.sum(gammaln([x + 1.]))

    result = - log_xfac - k * np.log(theta) - gammaln(k) + gammaln(k + x_sum) - (k + x_sum) * np.log(N + 1. / theta)

    return result if log else np.exp(result)


def nbin_evidence(x, a, b, N, r, log=False):
    x_sum = np.sum(x)

    result = betaln(a + N * r, b + x_sum) - betaln(a, b) + np.sum(np.log(scipy.special.binom(x + r - 1, x)))

    return result if log else np.exp(result)

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


def poisson_sum_evidence(x, k, theta, log=True):
    N = x.size
    sx = np.sum(x)

    result = -k * np.log(theta * N) - gammaln(k) - gammaln(sx + 1) + gammaln(k + sx) - (k + sx) * np.log(
        1 + 1. / (theta * N))

    return result if log else np.exp(result)


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

