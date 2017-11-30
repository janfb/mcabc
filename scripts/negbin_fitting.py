import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from random import shuffle
from scipy.stats import gamma, beta
import scipy.special

# The cost function of the MDN needs to evaluate the beta distribution written in pytorch terms:

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


def log_betafun(a, b):
    # beta function is defined in terms of gamma as: B(a, b) = gamma(a)gamma(b)/gamma(a + b)
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


def beta_pdf(mu, a, b, log=True):
    one = Variable(torch.ones(mu.size()).type_as(a.data))
    result = (a - 1) * torch.log(mu) + (b - 1) * torch.log(1 - mu) - log_betafun(a, b)
    return result if log else torch.exp(result)


def calculate_stats(x):
    sx = np.array([np.sum(x)])
    # sx = x
    return sx

def generate_dataset(N, m, alpha, beta, normalize=False):
    # N data sets
    # each with m samples

    X = []
    thetas = []

    for i in range(N):
        # sample from the prior
        mu = np.random.beta(alpha, beta)

        # generate samples

        x = np.random.negative_binomial(n=r, p=mu, size=m)

        # as data we append the summary stats
        X.append(calculate_stats(x).astype(float))
        thetas.append([mu])

    return X, np.array(thetas)

class MDN_phi(nn.Module):
    def __init__(self, ndim_input=1, ndim_output=1, n_hidden=5, n_components=1):
        super(MDN_phi, self).__init__()
        self.fc_in = nn.Linear(ndim_input, n_hidden)
        self.tanh = nn.Tanh()
        self.alpha_out = torch.nn.Sequential(
            nn.Linear(n_hidden, n_components),
            nn.Softplus())
        self.beta_out = torch.nn.Sequential(
            nn.Linear(n_hidden, n_components),
            nn.Softplus())

    def forward(self, x):
        out = self.fc_in(x)
        act = self.tanh(out)
        out_alpha = self.alpha_out(act)
        out_beta = self.beta_out(act)
        return (out_alpha, out_beta)


# the loss evaluates model (MoG) with the given data (y) and takes the log loss
def mdn_loss_function(out_shape, out_scale, y):
    result = beta_pdf(y, out_shape, out_scale, log=True)
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


def train(X, Y, model, optim, n_epochs=500, n_minibatch=50):
    dataset_train = [(x, y) for x, y in zip(X, Y)]

    losses = []

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

            losses.append(loss.data[0])

        if (epoch + 1) % 50 == 0:
            print("[epoch %04d] loss: %.4f" % (epoch + 1, loss.data[0]))
    return losses

def normalize(X, norm=None):
    if norm is None:
        xmean = X.mean(axis=0)
        xstd = X.std(axis=0)
    else:
        xmean = norm[0]
        xstd = norm[1]
    return (X - xmean) / xstd, (xmean, xstd)


n_samples = 100000
sample_size = 100

n_epochs = 500
n_minibatch = 500

# prior params
alp = 2.
bet = 2.

# NB param
true_mu = .4
r = 4

# the NB is defined for a fixed number of failures, r.
# It describes the distribution of failures until r successes have occured, or the other way around.
# thus, the number of trials is implicit
true_mean = r * true_mu / (1 - true_mu)

X, Y = generate_dataset(n_samples, sample_size, alpha=alp, beta=bet)
X, norm = normalize(np.array(X))

model = MDN_phi(ndim_input=1, n_components=1)
optim = torch.optim.Adam(model.parameters(), lr=0.01)

losses = train(X, Y, model, optim, n_epochs=n_epochs, n_minibatch=n_minibatch)


X_o = np.random.negative_binomial(n=r, p=true_mu, size=sample_size)
X_var = Variable(torch.Tensor(X_o.astype(float)))

stats_o = calculate_stats(X_o).astype(float).reshape(1, 1)
stats_o, norm = normalize(stats_o, norm)

X_var = Variable(torch.Tensor(stats_o))

(out_shape, out_scale) = model(X_var)

# plot posteriors
n_thetas = 1000
thetas = np.linspace(0.01, .99, n_thetas)

# predicted
posterior_pred = beta.pdf(x=thetas, a=out_shape.data.numpy(), b=out_scale.data.numpy()).squeeze()
prior = beta.pdf(x=thetas, a=alp, b=bet)

# analytical
a = alp + sample_size * r
b = bet + X_o.sum()
post_ana = beta.pdf(thetas, a, b)

plt.figure(figsize=(15, 5))
plt.plot(thetas, prior, label='beta prior')

plt.plot(thetas, posterior_pred, label='estimated posterior')
plt.plot(thetas, post_ana, label='true posterior')
plt.axvline(x=true_mu, ls='--', c='k', label=r'true $\mu$')
plt.xlabel(r'$\mu$')
plt.legend()

plt.savefig('../figures/negbin_fitting_N{}M{}.pdf'.format(n_samples, sample_size))
