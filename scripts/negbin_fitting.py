import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import scipy.special

from torch.autograd import Variable
from random import shuffle
from scipy.stats import gamma, beta

from model_comparison.helpers import *


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


def train(X, Y, model, optim, n_epochs=500, n_minibatch=50):
    dataset_train = [(x, y) for x, y in zip(X, Y)]

    losses = []

    for epoch in range(n_epochs):
        bgen = batch_generator(dataset_train, n_minibatch)

        for j, (x_batch, y_batch) in enumerate(bgen):
            x_var = Variable(torch.Tensor(x_batch))
            y_var = Variable(torch.Tensor(y_batch))

            (out_shape, out_scale) = model(x_var)
            loss = beta_mdn_loss(out_shape, out_scale, y_var)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.data[0])

        if (epoch + 1) % 50 == 0:
            print("[epoch %04d] loss: %.4f" % (epoch + 1, loss.data[0]))
    return losses


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
