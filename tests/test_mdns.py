import numpy as np
import scipy.stats
import torch

from model_comparison.utils.processing import generate_nd_gaussian_dataset, normalize, sample_poisson, \
    calculate_stats_toy_examples, sample_poisson_gamma_mixture
from model_comparison.mdn.MixtureDensityNetwork import MultivariateMogMDN, UnivariateMogMDN, ClassificationMDN
from model_comparison.mdn.Trainer import Trainer

from unittest import TestCase


class TestMDNs(TestCase):

    def test_posterior_fitting_with_mog(self):

        n_params = 2  # 2D problem, better visualization

        # define a MoG model with n_params + 1 inputs: data dimensions plus model index
        model = MultivariateMogMDN(ndim_input=n_params + 1, ndim_output=2)
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

        loss_trace = trainer.train(X, theta, n_epochs=10, n_minibatch=10)

    def test_posterior_fitting_univariate_mog(self):
        """
        Test with fitting a MoG to a posterior over the Poisson rate parameter of a Poisson model
        :return:
        """
        # set up conjugate Gamma prior
        gamma_prior = scipy.stats.gamma(a=2., scale=5.)
        # get data
        thetas, x = sample_poisson(gamma_prior, n_samples=100, sample_size=10)
        sx = calculate_stats_toy_examples(x)
        sx, norm = normalize(sx)

        # define a MoG model with n_params + 1 inputs: data dimensions plus model index
        model = UnivariateMogMDN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(model, optimizer, verbose=True)

        loss_trace = trainer.train(sx, thetas, n_epochs=10, n_minibatch=10)

    def test_classification_mdn(self):
        """
        Test the model comparison posterior approximation
        :return:
        """

        # set params
        sample_size = 10
        n_samples = 100

        # prior hyperparams
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
                theta, x = sample_poisson_gamma_mixture(prior_k, prior_theta, 1, sample_size)

            # calculate mean and var as summary stats
            X.append([np.mean(x), np.var(x)])
            thetas.append(theta)

        X = np.array(X)

        # normalize
        X, norm = normalize(X)

        # define a MoG model with n_params + 1 inputs: data dimensions plus model index
        model = ClassificationMDN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(model, optimizer, verbose=True, classification=True)

        loss_trace = trainer.train(X, m, n_epochs=10, n_minibatch=10)

    def test_multivariateMoGMDN_prediction(self):

        model = MultivariateMogMDN(ndim_input=2, ndim_output=2)

        post = model.predict(np.random.rand(1, 2))

    def test_transformation_to_delfi_posterior(self):

        model = MultivariateMogMDN(ndim_input=2, ndim_output=2)

        post = model.predict(np.random.rand(1, 2))

        post_dd = post.get_dd_object()

    def test_multivariate_get_dd_object(self):
        model = MultivariateMogMDN(ndim_input=2, ndim_output=2)

        pp = model.predict([[1., 1.]])
        dd = pp.get_dd_object()

        dd_means = [x.m.tolist() for x in dd.xs]

        pp_means = [pp.mus[:, :, k].data.numpy().squeeze().tolist() for k in range(pp.n_components)]

        assert dd_means == pp_means, 'means should be the same for every component'

        assert np.isclose(dd.mean, pp.mean).all(), 'over-all means should be equal: {}, {}'.format(dd.mean, pp.mean)

    def test_multivariate_get_quantile(self):

        model = MultivariateMogMDN(ndim_input=2, ndim_output=2, n_components=1)

        pp = model.predict([[1., 1.]])

        # just test the outer edges and the mean
        lower = pp.mean - 1e4
        upper = pp.mean + 1e4

        # get corresponding quantiles
        quantiles = pp.get_quantile(np.reshape([lower, upper], (2, -1)))

        assert np.isclose(quantiles, [0., 1.]).all(), 'quantiles should be close to [0, 1.], but {}'.format(quantiles)

    def test_multivariate_get_quantile_per_variable(self):

        model = MultivariateMogMDN(ndim_input=2, ndim_output=2, n_components=1)

        pp = model.predict([[1., 1.]])

        # just test the outer edges and the mean
        lower = pp.mean - 1e4
        upper = pp.mean + 1e4
        m = pp.mean

        quantiles = pp.get_quantile_per_variable(np.reshape([lower, upper, m], (3, -1)))

        assert np.isclose(quantiles, [[0., 0.], [1., 1.], [.5, .5]]).all(), 'incorrect quantiles: {}'.format(quantiles)

    def test_multivariate_gen(self):

        model = MultivariateMogMDN(ndim_input=2, ndim_output=2, n_components=3)

        pp = model.predict([[1., 1.]])

        # just test the outer edges and the mean
        ns = 10
        ss = pp.gen(ns)

        assert ss.shape == (ns, 2), 'samples shape have shape ({}, 2), have {}'.format(ns, ss.shape)
