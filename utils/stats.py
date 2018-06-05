import numpy as np
import scipy.stats
import tqdm

from scipy.special import gammaln, digamma, betaln


def calculate_pprob_from_evidences(pd1, pd2, priors=None):
    if priors is None:
        # p(m|d) = p(d | m) * p(m) / (sum_ p(d|m_i)p(m))))
        # because the prior is uniform we just return the normalized evidence:
        return pd1 / (pd1 + pd2)
    else:
        # p(m|d) = p(d | m) * p(m) / (sum_ p(d|m_i)p(m))))
        return pd1 * priors[0] / (pd1 * priors[0] + pd2 * priors[1])


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


def calculate_nb_evidence(x, k_k, theta_k, k_theta, theta_theta, log=False):
    # set up a grid of values around the priors
    # take grid over the whole range of the priors

    k_start = scipy.stats.gamma.ppf(1e-8, a=k_k)
    k_end = scipy.stats.gamma.ppf(1 - 1e-8, a=k_k)

    theta_start = scipy.stats.gamma.ppf(1e-8, a=k_theta)
    theta_end = scipy.stats.gamma.ppf(1 - 1e-8, a=k_theta)

    # set priors
    prior_k = scipy.stats.gamma(a=k_k, scale=theta_k)
    prior_theta = scipy.stats.gamma(a=k_theta, scale=theta_theta)

    (evidence, err) = scipy.integrate.dblquad(func=nb_evidence_integrant_direct,
                                              a=theta_start / (1 + theta_start),
                                              b=theta_end / (1 + theta_end),
                                              gfun=lambda x: k_start, hfun=lambda x: k_end,
                                              args=[x, prior_k, prior_theta])

    return np.log(evidence) if log else evidence

def nbinom_pmf(k, r, p):
    """
    Calculate pmf values according to Wikipedia definition of the negative binomial distribution:
    p(X=x | r, p = (x + r - 1)choose(x) p^x (1 - p)^r
    """

    return scipy.special.binom(k + r - 1, k) * np.power(p, k) * np.power(1-p, r)


def nb_evidence_integrant_direct(r, p, x, prior_k, prior_theta):
    """
    Negative Binomial marginal likelihood integrant: NB likelihood times prior pds values for given set of prior params
    """
    # set prior params for direct NB given params for indirect Poisson-Gamma mixture (Gamma priors on k and theta)

    # get pdf values
    pr = prior_k.pdf(r)
    # do change of variables or not?
    pp = np.power(1 - p, -2) * prior_theta.pdf(p / (1 - p))

    value = np.log(nbinom_pmf(x, r, p)).sum() + np.log(pr) + np.log(pp)

    return np.exp(value)


def calculate_pprob_from_evidences(pd1, pd2, priors=None):
    if priors is None:
        # p(m|d) = p(d | m) * p(m) / (sum_ p(d|m_i)p(m))))
        # because the prior is uniform we just return the normalized evidence:
        return pd1 / (pd1 + pd2)
    else:
        # p(m|d) = p(d | m) * p(m) / (sum_ p(d|m_i)p(m))))
        return pd1 * priors[0] / (pd1 * priors[0] + pd2 * priors[1])


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


def calculate_gamma_dkl(k1, theta1, k2, theta2):
    return (k1 - k2) * digamma(k1) - gammaln(k1) + gammaln(k2) + \
           k2 * (np.log(theta2) - np.log(theta1)) + k1 * (theta1 - theta2) / theta2


def calculate_dkl_1D_scipy(p_pdf_array, q_pdf_array):
    """
    Calculate DKL from array of pdf values.

    The arrays should cover as much of the range as possible.
    :param p_pdf_array:
    :param q_pdf_array:
    :return:
    """
    return scipy.stats.entropy(pk=p_pdf_array, qk=q_pdf_array)


def calculate_dkl_monte_carlo(x, p_pdf, q_pdf):
    """
    Estimate the DKL between 1D RV p and q.

    :param x: samples from p
    :param p_pdf: pdf function for p
    :param q_pdf: pdf function for q
    :return: estimate of dkl, standard error
    """

    # eval those under p and q
    pp = p_pdf(x)
    pq = q_pdf(x)

    # estimate expectation of log
    log = np.log(pp) - np.log(pq)
    dkl = log.mean()
    # estimate the standard error
    stderr = log.std(ddof=1) / np.sqrt(x.shape[0])

    return dkl, stderr


def calculate_dkl(p, q):
    """
    Calculate dkl between p and q.
    :param p: scipy stats object with .pdf() and .ppf() methods
    :param q: delfi.distribution object with .eval() method
    :return: dkl(p, q)
    """
    # parameter range
    p_start = p.ppf(1e-9)
    p_end = p.ppf(1 - 1e-9)

    # integral function
    def integrant(x):
        log_pp = p.logpdf(x)
        log_pq = q.eval(np.reshape(x, (1, -1)), log=True)
        return np.exp(log_pp) * (log_pp - log_pq)

    (dkl, err) = scipy.integrate.quad(integrant, a=p_start, b=p_end)
    return dkl


def calculate_credible_intervals_success(theta, ppf_fun, intervals, args=None):
    """
    Calculate credible intervals given a true parameter value and a percent point function of a distribution
    :param theta: true parameter
    :param ppf_fun: percent point function (inverse CDF)
    :param intervals: array-like, credible intervals to be calculated
    :param args: arguments to the ppf function
    :return: a binary vector, same length as intervals, indicating whether the true parameter lies in that interval
    """
    tails = (1 - intervals) / 2

    # get the boundaries of the credible intervals
    lows, highs = ppf_fun(tails, *args), ppf_fun(1 - tails, *args)
    success = np.ones_like(intervals) * np.logical_and(lows <= theta, theta <= highs)

    return success


def check_credible_regions(theta_o, cdf_fun, credible_regions):

    q = cdf_fun(theta_o)

    if q > 0.5:
        # the mass in the CR is 1 - how much mass is above times 2
        cr_mass = 1 - 2 * (1 - q)
    else:
        # or 1 - how much mass is below, times 2
        cr_mass = 1 - 2 * q
    counts = np.ones_like(credible_regions) * (credible_regions > cr_mass)
    return counts


def calculate_ppf_from_samples(qs, samples):
    """
    Given quantiles and samples, calculate values corresponding to the quantiles by approximating the
    MoG inverse CDF from samples.
    :param qs: quantiles, array-like
    :param samples: number of samples used to for sampling
    :return: corresponding values, array-like
    """

    qs = np.atleast_1d(qs)
    values = np.zeros_like(qs)

    # use bins from min to max
    bins = np.linspace(samples.min(), samples.max(), 1000)
    # asign samples to bins
    bin_idx = np.digitize(samples, bins)
    # count samples per bin --> histogram
    n = np.bincount(bin_idx.squeeze())
    # take the normalized cum sum as the cdf
    cdf = np.cumsum(n) / np.sum(n)

    # for every quantile, get the corresponding value on the cdf
    for i, qi in enumerate(qs):
        quantile_idx = np.where(cdf >= qi)[0][0]
        values[i] = bins[quantile_idx]

    return values


def inverse_transform_sampling_1d(array, pdf_array, n_samples):
    """
    Generate samples from an arbitrary 1D distribution given an array of pdf values. Using inverse transform sampling.

    Calculates CDF by summing up values in the PDF. Assumes values in array and PDF are spaced uniformly.

    :param array: array of RV values covering a representative range
    :param pdf_array: the corresponding PDF values of the values in 'array'.
    :param n_samples: number of samples to generate
    :return: array-like, array of pseudo-randomly generated sampled.
    """
    uniform_samples = scipy.stats.uniform.rvs(size=n_samples)
    samples = np.zeros(n_samples)
    # calculate the cdf by taking the cumsum and normaliying by dt
    cdf = np.cumsum(pdf_array) * (array[1] - array[0])

    for i, s in enumerate(uniform_samples):
        # find idx in cmf
        idx = np.where(cdf >= s)[0][0]
        # add the corresponding value
        samples[i] = array[idx]

    return samples


def inverse_transform_sampling_2d(x1, x2, joint_pdf, n_samples):
    """
    Generate samples from an arbitrary 2D distribution f(x, y) given a matrix of joint density values.

    Using 2D inverse transform sampling: Calculate the marginal p(x1) and the condition p(x2 | x1). Generate
    pseudo random samples from the x1 marginal. Then generate pseudo-random samples from the conditional, each sample
    conditioned on a x1 sample of the previos step.
    :param x1: values of RV x1
    :param x2: values of RV x2
    :param joint_pdf: 2D array of PDF values corresponding to the bins defined in x1 and x2
    :param n_samples: number of samples to draw
    :return: np array with samples (n_samples, 2)
    """

    # calculate marginal of x1 by integrating over x2
    x1_pdf = np.trapz(joint_pdf, x=x2, axis=1)

    # sample from marginal
    samples_x1 = inverse_transform_sampling_1d(x1, x1_pdf, n_samples)

    # calculate the conditional of x2 given x1 using Bayes rule
    # this gives a matrix of pdf, one for each values of x1 that we condition on.
    x2_pdf = np.zeros_like(joint_pdf)
    x2_cdf = np.zeros_like(joint_pdf)
    # condition on every x1
    for i in range(x1.size):
        # conditioned on this x1, apply Bayes
        px1 = x1_pdf[i] if x1_pdf[i] > 0. else 1e-12
        x2_pdf[i, ] = joint_pdf[i, :] / px1
        # get the corresponding cdf by cumsum and normalization
        x2_cdf[i, ] = np.cumsum(x2_pdf[i,])
        x2_cdf[i, ] /= np.max(x2_cdf[i,])
        assert np.isclose(x2_cdf[i, 0], 0, atol=1e-5), 'cdf should go from 0 to 1, {}'.format(x2_cdf[i, 0])
        assert np.isclose(x2_cdf[i, -1], 1, atol=1e-5), 'cdf should go from 0 to 1, {}'.format(x2_cdf[i, 0])

    # sample new uniform numbers
    uniform_samples = scipy.stats.uniform.rvs(size=n_samples)

    samples_x2 = []
    for uni_sample, x1_sample in zip(uniform_samples, samples_x1):
        # get the index of the x1 sample for conditioning
        idx_x1 = np.where(x1 >= x1_sample)[0][0]
        # find idx in conditional cmf
        idx_u = np.where(x2_cdf[idx_x1,] >= uni_sample)[0][0]

        # add the corresponding value
        samples_x2.append(x2[idx_u])

    return np.vstack((samples_x1, np.array(samples_x2))).T


class NBExactPosterior:
    """
    Class for the exact NB posterior. Defined by observed data and priors on k and theta, the shape and scale of the
    Gamma distribution in the Poisson-Gamma mixture.

    Has methods to calculate the exact posterior in terms of a joint pdf matrix using numerical integration.
    And methods to evaluate and to generate samples under this pdf.

    Once the posterior is calculated and samples are generated, it has properties mean and std to be compared to the
    predicted posterior.
    """

    def __init__(self, x, prior_k, prior_theta):
        """
        Instantiate the posterior with data and priors. the actual posterior has to be calculate using
        calculate_exact_posterior()
        :param x: observed data, array of counts
        :param prior_k: scipy.stats.gamma object
        :param prior_theta: scipy.stats.gamma object
        """

        # set flags
        self.samples_generated = False  # whether mean and std are defined
        self.calculated = False  # whether exact solution has been calculated

        self.xo = x
        self.prior_k = prior_k
        self.prior_th = prior_theta

        # prelocate
        self.evidence = None
        self.joint_pdf = None
        self.joint_cdf = None
        self.ks = None
        self.thetas = None

        self.samples = []

    def calculat_exact_posterior(self, theta_o, n_samples=200, prec=1e-6, verbose=True):
        """
        Calculate the exact posterior.
        :param theta_o: the true parameter theta
        :param n_samples: the number of entries per dimension on the joint_pdf grid
        :param prec: precision for the range of prior values
        :return: No return
        """

        # if not calculated
        if not self.calculated:
            self.calculated = True
            # set up a grid. take into account the true theta value to cover the region around it in the posterior
            # get the quantiles of the true theto under the prior
            k_pos = self.prior_k.cdf(theta_o[0])
            th_pos = self.prior_th.cdf(theta_o[1])

            # set the tail around it,
            tail = 0.8
            # choose ranges such that there are enough left and right of the true theta, use prec for bounds
            self.ks = np.linspace(self.prior_k.ppf(np.max((prec, k_pos - tail))),
                                  self.prior_k.ppf(np.min((1 - prec, k_pos + tail))), n_samples)
            self.thetas = np.linspace(self.prior_th.ppf(np.max((prec, th_pos - tail))),
                                      self.prior_th.ppf(np.min((1 - prec, th_pos + tail))), n_samples)

            joint_pdf = np.zeros((self.ks.size, self.thetas.size))

            # calculate likelihodd times prior for every grid value
            with tqdm.tqdm(total=self.ks.size * self.thetas.size, desc='calculating posterior',
                           disable=not verbose) as pbar:

                for i, k in enumerate(self.ks):
                    for j, th in enumerate(self.thetas):
                        r = k
                        p = th / (1 + th)
                        joint_pdf[i, j] = nb_evidence_integrant_direct(r, p, self.xo, self.prior_k, self.prior_th)
                        pbar.update()

            # calculate the evidence as the integral over the grid of likelihood * prior values
            self.evidence = np.trapz(np.trapz(joint_pdf, x=self.thetas, axis=1), x=self.ks, axis=0)
            self.joint_pdf = joint_pdf / self.evidence

            # calculate cdf
            # Calculate CDF by taking cumsum on each axis
            s1 = np.cumsum(np.cumsum(self.joint_pdf, axis=0), axis=1)
            # approximate cdf by summation and normalization
            self.joint_cdf = s1 / s1.max()
        else:
            print('already done')

    def eval(self, x, log=False):
        """
        Evaluate the joint pdf for value pairs given in x.
        :param x: np.array, shape (n, 2)
        :return: pdf values, np array, shape (n, )
        """

        x = np.atleast_1d(x)
        assert self.calculated, 'calculate the joint posterior first using calculate_exaxt_posterior'
        assert x.ndim == 2, 'x should have two dimensions, (n_samples, 2)'
        assert x.shape[1] == 2, 'each datum should have two entries, [k, theta]'

        pdf_values = []
        # for each pair of (k, theta)
        for xi in x:
            # look up indices in the ranges
            idx_k = np.where(self.ks >= xi[0])[0][0]
            idx_th = np.where(self.thetas >= xi[1])[0][0]

            # take corresponding pdf values from pdf grid
            pdf_values.append(self.joint_pdf[idx_k, idx_th])

        return np.log(np.array(pdf_values)) if log else np.array(pdf_values)

    # to mimic scipy.stats behavior
    def pdf(self, x):
        """
        Evaluate pdf at x
        :param x: samples
        :return: density values
        """
        return self.eval(x)

    def logpdf(self, x):
        """
        Evaluate log density at x
        :param x: samples
        :return: log density
        """
        return self.eval(x, log=True)

    def ppf(self, q):
        """
        Percent point function at q, or inverse CDF. Approximated by looking up the index in the cdf table
        that is closest to q.
        :param q: quantile
        :return: corresponding value on the RV range
        """
        q = np.atleast_1d(q)

        # look up the index of the quantile in the 2D CDF grid
        values = []
        for qi in q:
            # find index in grid for every dimension
            idx1, idx2 = np.where(self.joint_cdf >= qi)
            values.append([self.ks[idx1[0]], self.thetas[idx2[0]]])

        return np.array(values)

    def cdf(self, x):

        x = np.atleast_1d(x)
        qs = []

        for xi in x:
            # find idx of x on the cdf grid
            idx_k = np.where(self.ks >= xi[0])[0][0]
            idx_th = np.where(self.thetas >= xi[1])[0][0]

            # get value from cdf
            qs.append(self.joint_cdf[idx_k, idx_th])

        return np.array(qs)

    def gen(self, n_samples):
        """
        Generate samples under the joint pdf grid using inverse transform sampling
        :param n_samples:
        :return:
        """

        assert self.calculated, 'calculate the joint posterior first using calculate_exaxt_posterior'
        self.samples_generated = True

        # generate new samples
        samples = inverse_transform_sampling_2d(self.ks, self.thetas, self.joint_pdf, n_samples)

        # add to list of all samples
        self.samples += samples.tolist()

        return samples

    def rvs(self, n_samples):
        return self.gen(n_samples)

    @property
    def mean(self):
        if len(self.samples) == 0:
            self.gen(1000)
        return np.mean(self.samples, axis=0).reshape(-1)

    @property
    def std(self):
        if len(self.samples) == 0:
            self.gen(1000)
        return np.sqrt(np.diag(np.cov(np.array(self.samples).T))).reshape(-1)

    @property
    def cov(self):
        if len(self.samples) == 0:
            self.gen(1000)
        return np.cov(np.array(self.samples).T)

    def get_marginals(self):

        k_pdf = np.trapz(self.joint_pdf, x=self.thetas, axis=1)
        th_pdf = np.trapz(self.joint_pdf, x=self.ks, axis=0)

        return [Distribution(self.ks, k_pdf), Distribution(self.thetas, th_pdf)]


class Distribution:
    """
    Class for arbitrary distribution defined in terms of an array of pdf values. Used for representing the marginals
    of the numerically calculated NB posterior.
    """

    def __init__(self, support_array, pdf_array):

        self.support = support_array
        self.pdf_array = pdf_array

        self.cdf_array = np.cumsum(self.pdf_array)
        self.cdf_array /= self.cdf_array.max()
        self.samples = []

    def eval(self, x, log=False):

        pdf_values = []
        # for each sample
        for xi in x:
            # look up index in the supported range
            idx_i = np.where(self.support >= xi)[0][0]

            # take corresponding pdf value from pdf
            pdf_values.append(self.pdf_array[idx_i])

        return np.log(np.array(pdf_values)) if log else np.array(pdf_values)

    def pdf(self, x):
        return self.eval(x)

    def logpdf(self, x):
        return self.eval(x, log=True)

    def gen(self, n_samples):
        """
        Generate samples under the pdf using inverse transform sampling
        :param n_samples:
        :return: array-like, samples
        """
        # generate samples
        samples = inverse_transform_sampling_1d(self.support, self.pdf_array, n_samples=n_samples)
        # add to all samples
        self.samples += samples.tolist()

        return samples

    def ppf(self, qs):
        """
        Percent point function at q, or inverse CDF. Approximated by looking up the index in the cdf table
        that is closest to q.
        :param q: quantile
        :return: corresponding value on the RV range
        """
        q = np.atleast_1d(qs)

        # look up the index of the quantile in the 2D CDF grid
        values = []
        for q in qs:
            # find index in grid for every dimension
            idx1 = np.where(self.cdf_array >= q)[0][0]
            values.append(self.support[idx1])

        return np.array(values)

    def cdf(self, xs):
        """
        Evaluate CDF at every x in xs. Approximated by looking up the index in the cdf array.
        :param xs: RV values to evaluate
        :return: quantiles in [0, 1]
        """
        # make it an array in case it is a scalar.
        xs = np.atleast_1d(xs)

        cdf_values = []
        for xi in xs:
            # look up index in the support array
            idx = np.where(self.support >= xi)[0][0]
            # get the corresponding quantile
            cdf_values.append(self.cdf_array[idx])

        return np.array(cdf_values)

    def get_credible_interval_counts(self, th, credible_intervals):
        # get the quantile of theta

        q = self.cdf(th)

        # q mass lies below th, therefore the CI is
        if q > 0.5:
            # for q > .5, 1 - how much mass is above q times 2 (2 tails)
            ci = 1 - 2 * (1 - q)
        else:
            # how much mass is below, times 2 (2 tails)
            ci = 1 - 2 * q
        counts = np.ones_like(credible_intervals) * (credible_intervals >= ci)
        return counts

    @property
    def mean(self):
        """
        Mean estimated from samples
        :return:
        """
        if len(self.samples) == 0:
            self.gen(1000)

        return np.mean(self.samples)

    @property
    def std(self):
        """
        Mean estimated from samples
        :return:
        """
        if len(self.samples) == 0:
            self.gen(1000)

        return np.std(self.samples)


class JointGammaPrior:

    def __init__(self, prior_k, prior_theta):

        self.prior_k = prior_k
        self.prior_theta = prior_theta

    def gen(self, n_samples):

        sk = self.prior_k.rvs(n_samples)
        sth = self.prior_theta.rvs(n_samples)

        return np.vstack((sk, sth)).reshape(n_samples, 2)

    def pdf(self, samples):

        samples = np.atleast_1d(samples)
        assert samples.shape[1] == 2, 'samples should be (n_samples, 2)'

        pk = self.prior_k.pdf(samples[:, 0])
        pth = self.prior_theta.pdf(samples[:, 1])

        return pk * pth

    def rvs(self, n_samples):
        return self.gen(n_samples)

    def eval(self, samples):
        return self.pdf(samples)