import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from random import shuffle
from scipy.stats import gamma
import scipy.special


class MDN_phi(nn.Module):
    def __init__(self, ndim_input=1, ndim_output=1, n_hidden=5, n_components=1):
        super(MDN_phi, self).__init__()
        self.fc_in = nn.Linear(ndim_input, n_hidden)
        self.tanh = nn.Tanh()
        self.scale_out = torch.nn.Sequential(
            nn.Linear(n_hidden, n_components),
            nn.Softplus())
        self.shape_out = torch.nn.Sequential(
            nn.Linear(n_hidden, n_components),
            nn.Softplus())

    def forward(self, x):
        out = self.fc_in(x)
        act = self.tanh(out)
        out_shape = self.scale_out(act)
        out_scale = self.shape_out(act)
        return (out_shape, out_scale)


# the loss evaluates model (MoG) with the given data (y) and takes the log loss
def mdn_loss_function(out_shape, out_scale, y):
    result = gamma_pdf(y, out_shape, out_scale, log=True)
    result = torch.mean(result)  # mean over batch
    return -result


def batch_generator(dataset, batch_size=5):
    shuffle(dataset)
    N_full_batches = len(dataset) // batch_size
    for i in range(N_full_batches):
        idx_from = batch_size * i
        idx_to = batch_size * (i + 1)
        xs, ys = zip(*[(x, y) for x, y in dataset[idx_from:idx_to]])
        yield xs, ys


def train(X, Y, n_epochs=500, n_minibatch=50):
    dataset_train = [(x, y) for x, y in zip(X, Y)]

    for epoch in range(n_epochs):
        bgen = batch_generator(dataset_train, n_minibatch)

        for j, (x_batch, y_batch) in enumerate(bgen):
            x_var = Variable(torch.Tensor(x_batch))
            y_var = Variable(torch.Tensor(y_batch))

            (out_shape, out_scale) = model(x_var)
            loss = mdn_loss_function(out_shape, out_scale, y_var)

            optim.zero_grad()
            loss.backward()
            optim.step()

        if (epoch + 1) % 50 == 0:
            print("[epoch %04d] loss: %.4f" % (epoch + 1, loss.data[0]))


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
    beta = 1 / scale

    ll_1 = -beta * x
    ll_2 = (alpha - 1.0) * torch.log(x)
    ll_3 = alpha * torch.log(beta)
    ll_4 = -log_gamma(alpha)
    result = torch.sum(ll_1 + ll_2 + ll_3 + ll_4, -1)
    if log:
        return result
    else:
        return torch.exp(result)


def posterior_analytical(lam, x, alpha, beta, log=True):
    result = alpha * np.log(beta) - scipy.special.gammaln(alpha) - np.sum(scipy.special.gammaln(x + 1)) + (np.sum(
        x) + alpha - 1) * np.log(lam) - lam * (x.size + beta)
    return result if log else np.exp(result)


def poisson_evidence(x, a, b):
    x_sum = np.sum(x)
    log_xfac = np.sum(scipy.special.gammaln(x + 1))

    return a * np.log(b) - scipy.special.gammaln(a) - log_xfac + scipy.special.gammaln(a + x_sum) - (
                                                                                                    a + x_sum) * np.log(
        b + x.size)


# we need to define a generative model to generate samples (theta, x)
def generate_dataset(N, m):
    # N data sets
    # each with m samples

    X = []
    thetas = []

    for i in range(N):
        # sample from the prior
        lam = np.random.gamma(shape, scale)

        # generate samples
        x = np.random.poisson(lam=lam, size=m)

        # as data we append the summary stats
        X.append(calculate_stats(x).astype(float))
        thetas.append([lam])

    return X, np.array(thetas)


# calculate summary stats, for poisson this is just x, so for a vector it is sum x
def calculate_stats(x):
    sx = np.array([np.sum(x)])
    # sx = x
    return sx


def normalize(X, norm=None):
    if norm is None:
        xmean = X.mean(axis=0)
        xstd = X.std(axis=0)
    else:
        xmean = norm[0]
        xstd = norm[1]
    return (X - xmean) / xstd, (xmean, xstd)

sample_size = 100 # size of toy data
n_samples = 100000

n_epochs = 500
n_minibatch = 500

shape = 9.
scale = .5

true_lam = 4.

model = MDN_phi(ndim_input=1, n_components=1)
optim = torch.optim.Adam(model.parameters(), lr=0.01)

X, Y = generate_dataset(n_samples, sample_size)
X, norm = normalize(np.array(X))

train(X, Y, n_epochs=n_epochs, n_minibatch=n_minibatch)

# now evaluate the model at the observed data
X_o = np.random.poisson(lam=true_lam, size=sample_size)

X_var = Variable(torch.Tensor(X_o.astype(float)))

stats_o = calculate_stats(X_o).astype(float).reshape(1, 1)
stats_o, norm = normalize(stats_o, norm)

X_var = Variable(torch.Tensor(stats_o))

(out_shape, out_scale) = model(X_var)

n_thetas = 1000
thetas = np.linspace(2, 6, n_thetas)

post = gamma.pdf(x=thetas, a=out_shape.data.numpy(), scale=out_scale.data.numpy()).squeeze()
prior = gamma.pdf(x=thetas, a=shape, scale=scale)

# get true posterior

# first calculate the normalization factor, use log for stability
alpha = shape
beta = 1 / scale
evidence = poisson_evidence(X_o, alpha, beta)

# the the normalized posterior
true_post = posterior_analytical(thetas, X_o, alpha, beta, log=True) - evidence
#true_post = gamma.pdf(x=thetas, a=(shape + np.sum(X_o)), scale = 1. / (N + scale))

plt.figure(figsize=(15, 5))
plt.plot(thetas, post, label='estimated posterior')
plt.plot(thetas, prior, '--', label='gamma prior')
plt.plot(thetas, np.exp(true_post), label='true posterior given data')
plt.axvline(x=true_lam, label='true theta', linestyle='--', color='r')
plt.xlabel('theta')
plt.legend()

plt.savefig('../figures/poisson_fitting_N{}M{}.pdf'.format(n_samples, sample_size))

# # save results
# d = dict(model=model, post_pred=post, prior=prior, post_ana=true_post, xo=X_o, shape=shape, scale=scale)
# import pickle
# with o