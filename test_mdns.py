import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import scipy.stats

from torch.autograd import Variable
from random import shuffle

import sys
sys.path.append('../')
from model_comparison.utils import *
from model_comparison.mdns import Trainer, MultivariateMogMDN, ClassificationSingleLayerMDN

from unittest import TestCase


class TestMDNs(TestCase):

    def test_posterior_fitting_with_mog(self):

        n_params = 2  # 2D problem, better visualization

        # define a MoG model with n_params + 1 inputs: data dimensions plus model index
        model = MultivariateMogMDN(ndim_input=n_params + 1, ndim_output=2, n_hidden=20, n_components=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(model, optimizer, verbose=True)

        # use different priors on the mean
        prior1 = scipy.stats.multivariate_normal(mean=[0.5, 0.5], cov=np.eye(n_params))
        prior2 = scipy.stats.multivariate_normal(mean=[-0.5, -0.5], cov=np.eye(n_params))

        # use fixed covariance for both models
        data_cov = 0.5 * np.eye(n_params)

        n_samples = 100
        sample_size = 10

        X1, theta1 = generate_nd_gaussian_dataset(n_samples, sample_size, prior1, data_cov=data_cov)
        X2, theta2 = generate_nd_gaussian_dataset(n_samples, sample_size, prior2, data_cov=data_cov)

        X = np.vstack((np.hstack((X1, -1 * np.ones(n_samples).reshape(n_samples, 1))),
                       np.hstack((X2, np.ones(n_samples).reshape(n_samples, 1)))))
        X, training_norm = normalize(X)

        theta = np.vstack((theta1, theta2))

        loss_trace = trainer.train(X, theta, n_epochs=100, n_minibatch=10)

    def test_classification_mdn(self):

        sample_size = 10
        n_samples = 1000

        # set RNG
        seed = 2
        np.random.seed(seed)

        k1 = 9.0
        theta2 = 2.0
        k2 = 5.
        theta3 = 1.0
        k3 = 1

        # then the scale of the Gamma prior for the Poisson is given by
        theta1 = (k2 * theta2 * k3 * theta3) / k1

        # set the priors
        prior_lam = scipy.stats.gamma(a=k1, scale=theta1)
        prior_k = scipy.stats.gamma(a=k2, scale=theta2)
        prior_theta = scipy.stats.gamma(a=k3, scale=theta3)

        # generate a large data set for training

        X = []
        thetas = []
        m = []

        for sample_idx in range(n_samples):

            # sample model index
            m.append(int(np.round(np.random.rand())))

            if m[sample_idx] == 0:
                # sample poisson
                theta, x = sample_poisson(prior_lam, 1, sample_size)
            else:
                # sample poisson
                theta, x, lambs = sample_poisson_gamma_mixture(prior_k, prior_theta, 1, sample_size)

            # calculate mean and var as summary stats
            X.append([np.mean(x), np.var(x)])
            thetas.append(theta)

        X = np.array(X)

        # normalize
        X, norm = normalize(X)

        # define a MoG model with n_params + 1 inputs: data dimensions plus model index
        model = ClassificationSingleLayerMDN(ndim_input=2, n_hidden=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(model, optimizer, verbose=True, classification=True)

        loss_trace = trainer.train(X, m, n_epochs=100, n_minibatch=10)