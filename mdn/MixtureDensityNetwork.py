import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from model_comparison.mdn.PyTorchDistributions import PytorchMultivariateMoG, PytorchUnivariateMoG


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
        (idx1, idx2) = np.triu_indices(self.ndims)  # get indices of upper triangle, including diagonal
        U_mat[:, :, idx1, idx2] = U_vec.view(U_mat[:, :, idx1, idx2].size())  # assign vector elements to upper triangle
        # apply exponential to get positive diagonal
        (idx1, idx2) = np.diag_indices(self.ndims)  # get indices of diagonal elements
        U_mat[:, :, idx1, idx2] = torch.exp(U_mat[:, :, idx1, idx2])  # apply exponential to diagonal

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

