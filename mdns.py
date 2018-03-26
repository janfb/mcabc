import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import tqdm

import delfi.distribution as dd
from torch.autograd import Variable
from model_comparison.utils import *


class Trainer:

    def __init__(self, model, optimizer=None, classification=False, verbose=False):

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01) if optimizer is None else optimizer

        self.loss_trace = None

        self.verbose = verbose
        self.trained = False

        if classification:
            self.target_type = torch.LongTensor
        else:
            self.target_type = torch.Tensor

    def train(self, X, Y, n_epochs=500, n_minibatch=50):
        dataset_train = [(x, y) for x, y in zip(X, Y)]

        loss_trace = []

        with tqdm.tqdm(total=n_epochs, disable=not self.verbose,
                       desc='training') as pbar:
            for epoch in range(n_epochs):
                bgen = batch_generator(dataset_train, n_minibatch)

                for j, (x_batch, y_batch) in enumerate(bgen):
                    x_var = Variable(torch.Tensor(x_batch))
                    y_var = Variable(self.target_type(y_batch))

                    model_params = self.model(x_var)
                    loss = self.model.loss(model_params, y_var)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_trace.append(loss.data.numpy())
                pbar.update()

        self.loss_trace = loss_trace

        self.trained = True

        return loss_trace

    def predict(self, data, samples):

        raise NotImplementedError

        # assert self.trained, 'You need to train the network before predicting'
        # assert(isinstance(samples, Variable)), 'samples must be in torch Variable'
        # assert samples.size()[1] == self.model.ndims, 'samples must be 2D matrix with (batch_size, ndims)'
        # model_params = self.model(samples)


class PytorchUnivariateMoG:

    def __init__(self, mus, sigmas, alphas):

        assert isinstance(mus, Variable), 'all inputs need to be pytorch Variable objects'

        self.mus = mus
        self.sigmas = sigmas
        self.alphas = alphas

        self.nbatch, self.n_components = mus.size()

    def pdf(self, y, log=True):
        """
        Calculate the density values of a batch of variates given the corresponding mus, sigmas, alphas.
        Use log-sum-exp trick to improve numerical stability.

        return the (log)-probabilities of all the entries in the batch. Type: (n_batch, 1)-Tensor
        """

        n_data, n_components = self.mus.size()

        log_probs_mat = Variable(torch.zeros(n_data, n_components))

        # gather component log probs in matrix with components as columns, rows as data points
        for k in range(n_components):
            mu = self.mus[:, k].unsqueeze(1)
            sigma = self.sigmas[:, k].unsqueeze(1)
            lprobs = self.normal_pdf(y.unsqueeze(1), mu, sigma, log=True)
            log_probs_mat[:, k] = lprobs

        log_probs_batch = my_log_sum_exp(torch.log(self.alphas) + log_probs_mat, axis=1)

        if log:
            result = log_probs_batch
        else:
            result = torch.exp(log_probs_batch)

        return result

    def eval_numpy(self, samples, log=False):
        """

        :param samples: array-like in shape (1, n_samples)
        :param log: if true, log pdf are returned
        :return: pdf values, (1, n_samples)
        """
        # eval existing posterior for some params values and return pdf values in numpy format

        p_samples = np.zeros_like(samples)

        # for every component
        for k in range(self.n_components):
            alpha = self.alphas[0, k].data.numpy()
            mean = self.mus[0, k].data.numpy()
            sigma = self.sigmas[0, k].data.numpy()
            # add to result, weighted with alpha
            p_samples += alpha * scipy.stats.norm.pdf(x=samples, loc=mean, scale=sigma)

        if log:
            return np.log(samples)
        else:
            return p_samples

    def gen(self, n_samples):
        """
        Generate samples from the MoG.
        :param n_samples:
        :return:
        """

        # get the number of samples per component, according to mixture weights alpha
        ns = np.random.multinomial(n_samples, pvals=self.alphas.data.numpy().squeeze())

        # sample for each component
        samples = []
        for k, n in enumerate(ns):
            # construct scipy object
            mean = self.mus[0, k].data.numpy()
            sigma = self.sigmas[0, k].data.numpy()
            # add samples to list
            samples += scipy.stats.norm.rvs(loc=mean, scale=sigma, size=n).tolist()

        # shuffle and return
        np.random.shuffle(samples)

        return np.array(samples)

    @staticmethod
    def normal_pdf(y, mus, sigmas, log=True):
        result = -0.5 * torch.log(2 * np.pi * sigmas ** 2) - 1 / (2 * sigmas ** 2) * (y.expand_as(mus) - mus) ** 2
        if log:
            return result
        else:
            return torch.exp(result)

    def ppf(self, q):
        """
        Percent point function for univariate MoG: given a quantile / mass value, get the corresponding variable
        :param q: the quantile value, e.g., .95, .5 etc.
        :return: the parameter value, e.g., the value corresponding to q amount of mass
        """
        raise NotImplementedError()

    def calculate_credible_interval_counts(self, theta_o, intervals):
        """
        Credible interval count for theta_o.

        For a given theta_o and a list of credible intervals, check whether theta_o falls into the interval, for each
        interval. Return a binary vector indicating success
        :param theta_o: float
        :param intervals: np array
        :return: np array, shape as intervals.
        """

        # for each interval. intervals are defined in mass. we need the tails, left and right of the interval
        tails = (1 - intervals) / 2.

        # generate samples to approximate the inverse cdf
        samples = self.gen(10000)
        print(samples.shape)

        lows = calculate_ppf_from_samples(tails, samples)
        highs = calculate_ppf_from_samples(1 - tails, samples)
        counts = np.ones_like(intervals) * np.logical_and(lows <= theta_o, theta_o <= highs)

        return counts

    def get_quantile(self, x):
        """
        For sample(s) x calculate the corresponding quantiles
        :param x:
        :return:
        """
        # if x is a scalar, make it an array
        x = np.atleast_1d(x)
        # make sure x is 1D
        assert x.ndim == 1, 'the input samples have to be 1D'

        # the quantile of the MoG is the weighted sum of the quantiles of the Gaussians
        # for every component
        quantiles = np.zeros_like(x)
        for k in range(self.n_components):
            alpha = self.alphas[0, k].data.numpy()
            mean = self.mus[0, k].data.numpy()
            sigma = self.sigmas[0, k].data.numpy()

            # evaluate the inverse cdf for every component and add to
            # add, weighted with alpha
            quantiles += alpha * scipy.stats.norm.cdf(x=x, loc=mean, scale=sigma)

        return quantiles

    @property
    def mean(self):
        """
        Mean of MoG
        """
        m = 0

        for k in range(self.n_components):
            m += (self.alphas[:, k] * self.mus[:, k]).data.numpy().squeeze()

        return m

    @property
    def std(self):
        """
        Scale of MoG
        :return:
        """
        return np.sum([self.alphas[0][k] * self.sigmas[0][k] for k in range(self.n_components)]).data.numpy().squeeze()

    def get_dd_object(self):
        """
        Get the delfi.distribution object
        :return:
        """
        # convert to dd format
        a = self.alphas.data.numpy().squeeze().tolist()
        ms = [[m] for m in self.mus.data.numpy().squeeze().tolist()]
        Ss = [[[s ** 2]] for s in self.sigmas.data.numpy().squeeze().tolist()]

        # set up dd MoG object
        return dd.mixture.MoG(a=a, ms=ms, Ss=Ss)


class PytorchUnivariateGaussian:

    def __init__(self, mu, sigma):

        self.mu = mu
        self.sigma = sigma

    @property
    def mean(self):
        return self.mu.data.numpy()

    def eval(self, samples, log=False):
        """
        Calculate pdf values for given samples
        :param samples:
        :return:
        """
        result = -0.5 * torch.log(2 * np.pi * self.sigma ** 2) - \
                 1 / (2 * self.sigma ** 2) * (samples.expand_as(self.mu) - self.mu) ** 2
        if log:
            return result
        else:
            return torch.exp(result)

    def ppf(self, q):
        """
        Percent point function for univariate Gaussian
        """
        return scipy.stats.norm.ppf(q, loc=self.mu.data.numpy(), scale=self.sigma.data.numpy())

    def ztrans_inv(self, mean, std):

        m = std * self.mu + mean
        sigma = std * self.sigma

        return PytorchUnivariateGaussian(m, sigma)


class PytorchMultivariateMoG:

    def __init__(self, mus, Us, alphas):
        """
        Set up a MoG in PyTorch. ndims is the number of dimensions of the Gaussian
        :param mus: PyTorch Variable of shape (n_samples, ndims, ncomponents)
        :param Us: PyTorch Variable of shape (n_samples, ncomponents, ndims, ndims)
        :param alphas: PyTorch Variable of shape (n_samples, ncomponents)
        """

        assert isinstance(mus, Variable), 'all inputs need to be pytorch Variable objects'
        assert isinstance(Us, Variable), 'all inputs need to be pytorch Variable objects'
        assert isinstance(alphas, Variable), 'all inputs need to be pytorch Variable objects'

        self.mus = mus
        self.Us = Us
        self.alphas = alphas

        self.nbatch, self.ndims, self.n_components = mus.size()

    @property
    def mean(self):
        """
        Mean of the MoG
        """
        mean = 0
        for k in range(self.n_components):
            mean += (self.alphas[:, k] * self.mus[:, :, k]).data.numpy().squeeze()
        return mean

    def pdf(self, y, log=True):
        # get params: batch size N, ndims D, ncomponents K
        N, D, K = self.mus.size()

        # prelocate matrix for log probs of each Gaussian component
        log_probs_mat = Variable(torch.zeros(N, K))

        # take weighted sum over components to get log probs
        for k in range(K):
            log_probs_mat[:, k] = multivariate_normal_pdf(X=y, mus=self.mus[:, :, k], Us=self.Us[:, k, :, :], log=True)

        # now apply the log sum exp trick: sum_k alpha_k * N(Y|mu, sigma) = sum_k exp(log(alpha_k) + log(N(Y| mu, sigma)))
        # this give the log MoG density over the batch
        log_probs_batch = my_log_sum_exp(torch.log(self.alphas) + log_probs_mat, axis=1)  # sum over component axis=1

        # return log or linear density dependent on flag:
        if log:
            result = log_probs_batch
        else:
            result = torch.exp(log_probs_batch)

        return result

    def eval_numpy(self, samples):
        # eval existing posterior for some params values and return pdf values in numpy format

        p_samples = np.zeros(samples.shape)

        # for every component
        for k in range(self.n_components):
            alpha = self.alphas[0, k].data.numpy()
            mean = self.mus[0, :, k].data.numpy()
            U = self.Us[0, k,].data.numpy()
            # get cov from Choleski transform
            cov = np.linalg.inv(U.T.dot(U))
            # add to result, weighted with alpha
            p_samples += alpha * scipy.stats.multivariate_normal.pdf(x=samples, mean=mean, cov=cov)

        return p_samples

    def get_dd_object(self):
        """
        Get the delfi.distribution object
        :return: delfi.distribution.mixture.MoG object
        """

        a = []
        ms = []
        Us = []
        # for every component, add the alphas, means and Cholesky transform U of P to lists
        for k in range(self.n_components):
            a.append(self.alphas[:, k].data.numpy()[0])
            ms.append(self.mus[:, :, k].data.numpy().squeeze())
            Us.append(self.Us[:, k, :, :].data.numpy().squeeze())

        # delfi MoG takes lists over components as arguments
        return dd.mixture.MoG(a=a, ms=ms, Us=Us)

    def get_quantile(self, x):
        """
        For sample(s) x calculate the corresponding quantiles. Calculate quantiles of individual Gaussians using scipy
        and then take the weighted sum over components.
        :param x: shape (n_samples, ndims), at least (1, ndims)
        :return:
        """
        # if x is a scalar, make it an array
        x = np.atleast_1d(x)
        # make sure x is 1D
        assert x.ndim == 2, 'the input array should be 2D, (n_samples, ndims)'
        assert x.shape[1] == self.ndims, 'the number of entries per sample should be ndims={}'.format(self.ndims)

        # the quantile of the MoG is the weighted sum of the quantiles of the Gaussians
        # for every component
        quantiles = np.zeros(x.shape[0])
        for k in range(self.n_components):
            alpha = self.alphas[:, k].data.numpy()[0]
            mean = self.mus[:, :, k].data.numpy().squeeze()
            U = self.Us[:, k, :, :].data.numpy().squeeze()
            # get cov from Choleski transform
            C = np.linalg.inv(U.T)
            S = np.dot(C.T, C)
            # add to result, weighted with alpha
            quantiles += alpha * scipy.stats.multivariate_normal.cdf(x=x, mean=mean, cov=S)

        return quantiles

    def get_quantile_per_variable(self, x):
        """
        Calculate the quantile of each parameter component in x, under the corresponding marginal of that component.

        :param x: (n_samples, ndims), ndims is the number of variables the MoG is defined for, e.g., k and theta,
        :return: quantile for every sample and for every variable of the MoG, (n_samples, ndims).
        """
        # for each variable, get the marginal and take the quantile weighted over components

        quantiles = np.zeros_like(x)

        for k in range(self.n_components):
            alpha = self.alphas[:, k].data.numpy()[0]
            mean = self.mus[:, :, k].data.numpy().squeeze()
            U = self.Us[:, k, :, :].data.numpy().squeeze()
            # get cov from Choleski transform
            C = np.linalg.inv(U.T)
            # covariance matrix
            S = np.dot(C.T, C)

            # for each variable
            for vi in range(self.ndims):
                # the marginal is a univariate Gaussian with the sub mean and covariance
                marginal = scipy.stats.norm(loc=mean[vi], scale=np.sqrt(S[vi, vi]))
                # the quantile under the marginal of vi for this component, for all n_samples
                q = marginal.cdf(x=x[:, vi])
                # take sum, weighted with component weight alpha
                quantiles[:, vi] += alpha * q

        return quantiles

    def get_marginals(self):
        """
        Return a list of PytorchUnivariateMoG holding the marginals of this PytorchMultivariateMoG.
        :return: list
        """
        assert self.nbatch == 1, 'this defined only for a single data point MoG'
        sigmas = np.zeros((self.ndims, self.n_components))
        # get sigma for every component
        for k in range(self.n_components):
            U = self.Us[:, k, :, :].data.numpy().squeeze()
            # get cov from Choleski transform
            C = np.linalg.inv(U.T)
            # covariance matrix
            S = np.dot(C.T, C)
            # the diagonal element is the variance of each variable, take sqrt to get std.
            sigmas[:, k] = np.sqrt(np.diag(S))

        # for each variable
        marginals = []
        for vi in range(self.ndims):
            # take the corresponding mean component, the sigma component extracted above. for all MoG compoments.
            m = self.mus[:, vi, :]
            std = Variable(torch.Tensor(sigmas[vi,].reshape(1, -1)))
            marg = PytorchUnivariateMoG(mus=m, sigmas=std, alphas=self.alphas)
            marginals.append(marg)

        return marginals

    def gen(self, n_samples):
        """
        Generate samples from the MoG.
        :param n_samples:
        :return:
        """

        # get the number of samples per component, according to mixture weights alpha
        ns = np.random.multinomial(n_samples, pvals=self.alphas.data.numpy().squeeze())

        # sample for each component
        samples = []
        for k, n in enumerate(ns):
            # construct scipy object
            mean = self.mus[:, :, k].data.numpy().squeeze()
            U = self.Us[:, k, :, :].data.numpy().squeeze()
            # get cov from Choleski transform
            C = np.linalg.inv(U.T)
            S = np.dot(C.T, C)

            # add samples to list
            samples += scipy.stats.multivariate_normal.rvs(mean=mean, cov=S, size=n).tolist()

        # shuffle and return
        np.random.shuffle(samples)

        return samples



class UnivariateMogMDN(nn.Module):

    def __init__(self, ndim_input=2, n_hidden=5, n_components=3):
        super(UnivariateMogMDN, self).__init__()
        self.fc_in = nn.Linear(ndim_input, n_hidden)
        self.tanh = nn.Tanh()
        self.alpha_out = torch.nn.Sequential(
              nn.Linear(n_hidden, n_components),
              nn.Softmax(dim=1)
            )
        self.logsigma_out = nn.Linear(n_hidden, n_components)
        self.mu_out = nn.Linear(n_hidden, n_components)

    def forward(self, x):
        # make sure the first dimension is at least singleton
        assert x.dim() >= 2
        out = self.fc_in(x)
        act = self.tanh(out)
        out_alpha = self.alpha_out(act)
        out_sigma = torch.exp(self.logsigma_out(act))
        out_mu = self.mu_out(act)
        return out_mu, out_sigma, out_alpha

    def loss(self, model_params, y):
        out_mu, out_sigma, out_alpha = model_params

        batch_mog = PytorchUnivariateMoG(mus=out_mu, sigmas=out_sigma, alphas=out_alpha)
        result = batch_mog.pdf(y, log=True)

        result = torch.mean(result)  # mean over batch

        return -result

    def predict(self, sx):
        """
        Take input sx and predict the corresponding posterior over parameters
        :param sx: shape (n_samples, n_features), e.g., for single sx (1, n_stats)
        :return: pytorch univariate MoG
        """
        if not isinstance(sx, Variable):
            sx = Variable(torch.Tensor(sx))

        assert sx.dim() == 2, 'the input should be 2D: (n_samples, n_features)'

        out_mu, out_sigma, out_alpha = self.forward(sx)

        return PytorchUnivariateMoG(out_mu, out_sigma, out_alpha)


class MultivariateMogMDN(nn.Module):

    def __init__(self, ndim_input=3, ndim_output=2, n_hidden_units=10, n_hidden_layers=1, n_components=3):
        super(MultivariateMogMDN, self).__init__()

        # dimensionality of the Gaussian components
        self.ndims = ndim_output
        self.n_components = n_components
        # the number of entries in the upper triangular Choleski transform matrix of the precision matrix
        self.utriu_entries = int(self.ndims * (self.ndims - 1) / 2) + self.ndims

        # activation
        self.activation_fun = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # input layer
        self.input_layer = nn.Linear(ndim_input, n_hidden_units)

        # add a list of hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Linear(n_hidden_units, n_hidden_units))

        # output layer maps to different output vectors
        # the mean estimates
        self.output_mu = nn.Linear(n_hidden_units, ndim_output * n_components)

        # output layer to precision estimates
        # the upper triangular matrix for D-dim Gaussian has m = (D**2 + D) / 2 entries
        # this should be a m-vector for every component. currently it is just a scalar for every component.
        # or it could be a long vector of length m * k, i.e, all the k vector stacked.
        self.output_layer_U = nn.Linear(n_hidden_units, self.utriu_entries * n_components)

        # additionally we have the mixture weights alpha
        self.output_layer_alpha = nn.Linear(n_hidden_units, n_components)

    def forward(self, x):
        batch_size = x.size()[0]

        # input layer
        x = self.activation_fun(self.input_layer(x))

        # hidden layers
        for layer in self.hidden_layers:
            x = self.activation_fun(layer(x))

        # get mu output
        out_mu = self.output_mu(x)
        out_mu = out_mu.view(batch_size, self.ndims, self.n_components)

        # get alpha output
        out_alpha = self.softmax(self.output_layer_alpha(x))

        # get activate of upper triangle U vector
        U_vec = self.output_layer_U(x)
        # prelocate U matrix
        U_mat = Variable(torch.zeros(batch_size, self.n_components, self.ndims, self.ndims))

        # assign vector to upper triangle of U
        (idx1, idx2) = np.triu_indices(self.ndims)
        U_mat[:, :, idx1, idx2] = U_vec
        # apply exponential to get positive diagonal
        (idx1, idx2) = np.diag_indices(self.ndims)
        U_mat[:, :, idx1, idx2] = torch.exp(U_mat[:, :, idx1, idx2])

        return out_mu, U_mat, out_alpha

    def loss(self, model_params, y):

        mu, U, alpha = model_params

        batch_mog = PytorchMultivariateMoG(mu, U, alpha)
        result = batch_mog.pdf(y, log=True)

        result = torch.mean(result)  # mean over batch

        return -result

    def predict(self, sx):
        """
        Take input sx and predict the corresponding posterior over parameters
        :param sx: shape (n_samples, n_features), e.g., for single sx (1, n_stats)
        :return: pytorch univariate MoG
        """
        if not isinstance(sx, Variable):
            sx = Variable(torch.Tensor(sx))

        assert sx.dim() == 2, 'the input should be 2D: (n_samples, n_features)'

        out_mu, U_mat, out_alpha = self.forward(sx)

        return PytorchMultivariateMoG(out_mu, U_mat, out_alpha)


class ClassificationMDN(nn.Module):

    def __init__(self, n_input=2, n_output=2, n_hidden_units=10, n_hidden_layers=1):
        super(ClassificationMDN, self).__init__()

        self.n_hidden_layers = n_hidden_layers

        # define funs
        self.softmax = nn.Softmax(dim=1)
        self.activation_fun = nn.Tanh()
        self.loss = nn.CrossEntropyLoss()

        # define architecture
        # input layer takes features, expands to n_hidden units
        self.input_layer = nn.Linear(n_input, n_hidden_units)
        # middle layer takes activates, passes fully connected to next layer
        # take arbitrary number of hidden layers:
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.n_hidden_layers):
            self.hidden_layers.append(nn.Linear(n_hidden_units, n_hidden_units))

        # last layer takes activation, maps to output vectors, applies activation function and softmax for normalization
        self.output_layer = nn.Linear(n_hidden_units, n_output)

    def forward(self, x):
        assert x.dim() == 2
        # batch_size, n_features = x.size()

        # forward path
        # input layer, pass x and calculate activations
        x = self.activation_fun(self.input_layer(x))

        # iterate n hidden layers, input x and calculate tanh activation
        for layer in self.hidden_layers:
            x = self.activation_fun(layer(x))

        # in the last layer, apply softmax
        p_hat = self.softmax(self.output_layer(x))

        return p_hat

    def predict(self, x):
        if not isinstance(x, Variable):
            x = Variable(torch.Tensor(x))

        assert x.dim() == 2, 'the input should be 2D: (n_samples, n_features)'

        p_vec = self.forward(x)


        return p_vec.data.numpy().squeeze()


# keep this class for backwards compability.
class ClassificationSingleLayerMDN(nn.Module):

    def __init__(self, ndim_input=2, ndim_output=2, n_hidden=5):
        super(ClassificationSingleLayerMDN, self).__init__()

        self.fc_in = nn.Linear(ndim_input, n_hidden)
        self.tanh = nn.Tanh()
        self.m_out = nn.Linear(n_hidden, ndim_output)

        self.loss = nn.CrossEntropyLoss()
        # the softmax is taken over the second dimension, given that the input x is (n_samples, n_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # make sure x has n samples in rows and features in columns: x is (n_samples, n_features)
        assert x.dim() == 2, 'the input should be 2D: (n_samples, n_features)'
        out = self.fc_in(x)
        act = self.tanh(out)
        out_m = self.softmax(self.m_out(act))

        return out_m

    def predict(self, sx):
        if not isinstance(sx, Variable):
            sx = Variable(torch.Tensor(sx))

        assert sx.dim() == 2, 'the input should be 2D: (n_samples, n_features)'

        p_vec = self.forward(sx)

        return p_vec.data.numpy().squeeze()
