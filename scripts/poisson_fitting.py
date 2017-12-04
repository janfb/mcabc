import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from random import shuffle
from scipy.stats import gamma
import scipy.special

from model_comparison.utils import *

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


def train(X, Y, n_epochs=500, n_minibatch=50):
    dataset_train = [(x, y) for x, y in zip(X, Y)]

    for epoch in range(n_epochs):
        bgen = batch_generator(dataset_train, n_minibatch)

        for j, (x_batch, y_batch) in enumerate(bgen):
            x_var = Variable(torch.Tensor(x_batch))
            y_var = Variable(torch.Tensor(y_batch))

            (out_shape, out_scale) = model(x_var)
            loss = gamma_mdn_loss(out_shape, out_scale, y_var)

            optim.zero_grad()
            loss.backward()
            optim.step()

        if (epoch + 1) % 50 == 0:
            print("[epoch %04d] loss: %.4f" % (epoch + 1, loss.data[0]))

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


sample_size = 100 # size of toy data
n_samples = 200000

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
true_post = gamma.pdf(x=thetas, a=(shape + np.sum(X_o)), scale = 1. / (sample_size + scale))

plt.figure(figsize=(15, 5))
plt.plot(thetas, post, label='estimated posterior')
plt.plot(thetas, prior, '--', label='gamma prior')
plt.plot(thetas, true_post, label='true posterior given data')
plt.axvline(x=true_lam, label='true theta', linestyle='--', color='r')
plt.xlabel('theta')
plt.legend()

plt.savefig('../figures/poisson_fitting_N{}M{}.pdf'.format(n_samples, sample_size))

# # save results
# d = dict(model=model, post_pred=post, prior=prior, post_ana=true_post, xo=X_o, shape=shape, scale=scale)
# import pickle
# with o