import delfi.distribution
import numpy as np
import scipy
import torch

from torch.autograd import Variable


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
            log_probs_mat[:, k] = lprobs.squeeze()

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

    def get_credible_interval_counts(self, theta_o, intervals):
        """
        Count whether a parameter falls in different credible intervals.

        Counting is done without sampling. Just look up the quantile q of theta. Then q mass lies below theta. If q is
        smaller than 0.5, then this is a tail and 1 - 2*tail is the CI. If q is greater than 0.5, then 1 - q is a tail
        and 1 - 2*tail is the CI.
        :param theta_o: parameter for which to calculate the CI counts, float
        :param intervals: np array
        :return: np array of {0, 1} for counts
        """
        # get the quantile of theta
        q = self.get_quantile(theta_o)

        # q mass lies below th, therefore the CI is
        if q > 0.5:
            # for q > .5, 1 - how much mass is above q times 2 (2 tails)
            ci = 1 - 2 * (1 - q)
        else:
            # how much mass is below, times 2 (2 tails)
            ci = 1 - 2 * q
        counts = np.ones_like(intervals) * (intervals>= ci)
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
        Scale of MoG. Formular from
        https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        :return:
        """
        a = self.alphas[0, :].data.numpy()
        vars = self.sigmas[0, :].data.numpy()**2
        ms = self.mus[0, :].data.numpy()

        var = np.sum([a[k] * (vars[k] + ms[k]**2) for k in range(self.n_components)]) - \
              np.sum([a[k] * ms[k] for k in range(self.n_components)])**2

        return np.sqrt(var)

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
        return delfi.distribution.mixture.MoG(a=a, ms=ms, Ss=Ss)

    def ztrans_inv(self, mean, std):
        """
        Apply inverse z transform.
        :param mean: original mean
        :param std: original std
        :return: PytorchUnivariateMoG with transformed means and stds
        """

        # apply same transform to every component
        new_mus = self.mus * std + mean
        new_sigmas = self.sigmas * std

        return PytorchUnivariateMoG(new_mus, new_sigmas, self.alphas)


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

        # prelocate covariance matrix for later calculation
        self.Ss = None

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

    @property
    def std(self):

        if self.Ss is None:
            Ss = self.get_Ss_from_Us()

        S = self.get_covariance_matrix()

        return np.sqrt(np.diag(S))

    def get_covariance_matrix(self):
        """
        Calculate the overall covariance of the MoG.

        The covariance of a set of RVs is the mean of the conditional covariances plus the covariances of
        the conditional means.
        The MoG is a weighted sum of Gaussian RVs. Therefore, the mean covariance are just the weighted sum of
        component covariances, similar for the conditional means. See here for an explanantion:

        https://math.stackexchange.com/questions/195911/covariance-of-gaussian-mixtures

        :return: Overall covariance matrix
        """
        if self.Ss is None:
            _ = self.get_Ss_from_Us()

        assert self.nbatch == 1, 'covariance matrix is returned only for single batch sample, but ' \
                                 'self.nbatch={}'.format(self.nbatch)

        # assume single batch sample
        batch_idx = 0
        S = np.zeros((self.ndims, self.ndims))

        a = self.alphas[batch_idx, :].data.numpy().squeeze()
        mus = self.mus[batch_idx, :, :].data.numpy().squeeze()
        ss = self.Ss[batch_idx, :, :, :].squeeze()

        m = np.dot(a, mus.T)

        # get covariance shifted by the means, weighted with alpha
        for k in range(self.n_components):
            S += a[k] * (ss[k, :, :] + np.outer(mus[:, k], mus[:, k]))

        # subtract weighted means
        S -= np.outer(m, m)

        return S

    def pdf(self, y, log=True):
        # get params: batch size N, ndims D, ncomponents K
        N, D, K = self.mus.size()

        # prelocate matrix for log probs of each Gaussian component
        log_probs_mat = Variable(torch.zeros(N, K))

        # take weighted sum over components to get log probs
        for k in range(K):
            log_probs_mat[:, k] = multivariate_normal_pdf(X=y, mus=self.mus[:, :, k], Us=self.Us[:, k, :, :],
                                                          log=True).squeeze()

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

        p_samples = np.zeros(samples.shape[:-1])

        # for every component
        for k in range(self.n_components):
            alpha = self.alphas[:, k].data.numpy()[0]
            mean = self.mus[:, :, k].data.numpy().squeeze()
            U = self.Us[:, k, :, :].data.numpy().squeeze()
            # get cov from Choleski transform
            C = np.linalg.inv(U).T
            S = np.dot(C.T, C)
            # add to result, weighted with alpha
            p_samples += alpha * scipy.stats.multivariate_normal.pdf(x=samples, mean=mean, cov=S)

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
        return delfi.distribution.mixture.MoG(a=a, ms=ms, Us=Us)

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

    def check_credible_regions(self, theta_o, credible_regions):
        """
        Count whether a parameter falls in different credible regions.

        Counting is done without sampling. Just look up the quantile q of theta. Then q mass lies below theta. If q is
        smaller than 0.5, then this is a tail and 1 - 2*tail is the CR. If q is greater than 0.5, then 1 - q is a tail
        and 1 - 2*tail is the CR.
        :param theta_o: parameter for which to calculate the CR counts, float
        :param credible_regions: np array of masses that define the CR
        :return: np array of {0, 1} for counts
        """

        q = self.get_quantile(theta_o.reshape(1, -1))
        if q > 0.5:
            # the mass in the CR is 1 - how much mass is above times 2
            cr_mass = 1 - 2 * (1 - q)
        else:
            # or 1 - how much mass is below, times 2
            cr_mass = 1 - 2 * q
        counts = np.ones_like(credible_regions) * (credible_regions > cr_mass)
        return counts

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
            std = Variable(torch.Tensor(sigmas[vi, ].reshape(1, -1)))
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
        ps = np.atleast_1d(self.alphas.data.numpy().squeeze())
        ns = np.random.multinomial(n_samples, pvals=ps)

        # sample for each component
        samples = np.zeros((1, 2)) # hack for initialization
        lower = 0
        for k, n in enumerate(ns):
            # construct scipy object
            mean = self.mus[:, :, k].data.numpy().squeeze()
            U = self.Us[:, k, :, :].data.numpy().squeeze()
            # get cov from Choleski transform
            C = np.linalg.inv(U.T)
            S = np.dot(C.T, C)

            # add samples to list
            ss = np.atleast_2d(scipy.stats.multivariate_normal.rvs(mean=mean, cov=S, size=n))
            samples = np.vstack((samples, ss))

        # remove hack
        samples = samples[1:, :]
        # shuffle and return
        np.random.shuffle(samples)

        return samples

    def get_Ss_from_Us(self):
        """
        Get the covariance from the matrix of Cholesky transforms of the precision matrices.
        :return:
        """

        # prelocate
        Ss = np.zeros_like(self.Us.data.numpy())

        # loop over batch
        for d in range(self.nbatch):
            # loop over components
            for k in range(self.n_components):
                # get the U matrix
                U = self.Us[d, k, ].data.numpy()
                # inverse the Cholesky transform to that of the covariance matrix
                C = np.linalg.inv(U.T)
                # get the covariance matrix from its Cholesky transform
                Ss[d, k, ] = np.dot(C.T, C)

        # set the matrix as attribute
        self.Ss = Ss

        return Ss

    def ztrans_inv(self, mean, std):
        """
        Inverse ztransform.

        Given a mean and std used for ztransform, return the PytorchMultivariateMoG holding the original location and
        scale. Assumes that the current loc and scale of the indivudual Gaussian is close to 0, 1, i.e., that the
        covariance matrix is a diagonal matrix.
        Applies the transform to every component separately and keeps the alpha for the new MoG.

        :param mean: mean of the original distribution
        :param std: vector of standard deviations of the original distribution
        :return: PytorchMultivariateMoG object with the original mean and variance.
        """

        mus = np.zeros((self.nbatch, self.ndims, self.n_components))
        Us = np.zeros((self.nbatch, self.n_components, self.ndims, self.ndims))

        Ssz = self.get_Ss_from_Us()

        # for every component
        for d in range(self.nbatch):
            for k in range(self.n_components):
                mus[d, :, k] = std * self.mus[d, :, k].data.numpy() + mean
                S = np.outer(std, std) * Ssz[d, k, ]
                Sin = np.linalg.inv(S)
                U = np.linalg.cholesky(Sin).T
                Us[d, k,] = U

        return PytorchMultivariateMoG(Variable(torch.Tensor(mus.tolist())),
                                      Variable(torch.Tensor(Us.tolist())), self.alphas)


def my_log_sum_exp(x, axis=None):
    """
    Apply log-sum-exp with subtraction of the largest element to improve numerical stability.
    """
    (x_max, idx) = torch.max(x, dim=axis, keepdim=True)

    return torch.log(torch.sum(torch.exp(x - x_max), dim=axis, keepdim=True)) + x_max


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