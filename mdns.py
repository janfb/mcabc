import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from utils import multivariate_normal_pdf, my_log_sum_exp, batch_generator
import scipy.stats


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

            if (epoch + 1) % 100 == 0 and self.verbose:
                print("[epoch %04d] loss: %.4f" % (epoch + 1, loss.data[0]))

        self.loss_trace = loss_trace

        self.trained = True

        return loss_trace

    def predict(self, data, samples):

        assert self.trained, 'You need to train the network before predicting'
        assert(isinstance(samples, Variable)), 'samples must be in torch Variable'
        assert samples.size()[1] == self.model.ndims, 'samples must be 2D matrix with (batch_size, ndims)'

        model_params = self.model(samples)


class PytorchMultivariateMoG:

    def __init__(self, mus, Us, alphas):

        assert isinstance(mus, Variable), 'all inputs need to be pytorch Variable objects'

        self.mus = mus
        self.Us = Us
        self.alphas = alphas

        self.nbatch, self.ndims, self.n_components = mus.size()

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

        sample_shape = samples.shape[:-1]
        p_samples = np.zeros(sample_shape)

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

    def rvs(self):
        pass


class MultivariateMogMDN(nn.Module):

    def __init__(self, ndim_input=3, ndim_output=2, n_hidden=10, n_components=3):
        super(MultivariateMogMDN, self).__init__()

        # dimensionality of the Gaussian components
        self.ndims = ndim_output
        self.n_components = n_components
        # the number of entries in the upper triangular Choleski transform matrix of the precision matrix
        self.utriu_entries = int(self.ndims * (self.ndims - 1) / 2) + self.ndims

        # input layer
        self.fc_in = nn.Linear(ndim_input, n_hidden)
        # activation
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # output layer the mean estimates
        self.mu_out = nn.Linear(n_hidden, ndim_output * n_components)

        # output layer to precision estimates
        # the upper triangular matrix for D-dim Gaussian has m = (D**2 + D) / 2 entries
        # this should be a m-vector for every component. currently it is just a scalar for every component.
        # or it could be a long vector of length m * k, i.e, all the k vector stacked.
        self.U_out = nn.Linear(n_hidden, self.utriu_entries * n_components)

        # additionally we have the mixture weights alpha
        self.alpha_out = nn.Linear(n_hidden, n_components)

    def forward(self, x):
        batch_size = x.size()[0]

        out = self.fc_in(x)
        act = self.tanh(out)

        out_mu = self.mu_out(act)
        out_mu = out_mu.view(batch_size, self.ndims, self.n_components)
        out_alpha = self.softmax(self.alpha_out(act))

        # get activate of upper triangle U vector
        U_vec = self.U_out(act)
        # prelocate U matrix
        U_mat = Variable(torch.zeros(batch_size, self.n_components, self.ndims, self.ndims))

        # assign vector to upper triangle of U
        (idx1, idx2) = np.triu_indices(self.ndims)
        U_mat[:, :, idx1, idx2] = U_vec
        # apply exponential to get positive diagonal
        (idx1, idx2) = np.diag_indices(self.ndims)
        U_mat[:, :, idx1, idx2] = torch.exp(U_mat[:, :, idx1, idx2])

        return (out_mu, U_mat, out_alpha)

    def loss(self, model_params, y):

        mu, U, alpha = model_params

        batch_mog = PytorchMultivariateMoG(mu, U, alpha)
        result = batch_mog.pdf(y, log=True)

        result = torch.mean(result)  # mean over batch

        return -result


class ClassificationSingleLayerMDN(nn.Module):

    def __init__(self, ndim_input=2, ndim_output=2, n_hidden=5):
        super(ClassificationSingleLayerMDN, self).__init__()
        self.fc_in = nn.Linear(ndim_input, n_hidden)
        self.tanh = nn.Tanh()
        self.m_out = nn.Linear(n_hidden, ndim_output)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.fc_in(x)
        act = self.tanh(out)
        out_m = self.m_out(act)

        return out_m
