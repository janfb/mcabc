import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from random import shuffle
from scipy.stats import gamma
import scipy
from scipy.special import gammaln

from model_comparison.utils import *


def generate_dataset(n_samples, sample_size):
    X = []
    thetas = []
    m = []

    # for every sample we want a triplet (m_i, theta, sx)
    for i in range(n_samples):
        # sample model index
        m_i = np.round(np.random.rand()).astype(int)

        # generate data from model
        # m_i in {0, 1} just sets the index in the array of prior hyperparams
        shape = shapes[m_i]
        scale = scales[m_i]
        theta, x = generate_poisson(sample_size, shape, scale)
        sx = calculate_stats(x)

        X.append([sx])
        thetas.append([theta])
        m.append([int(m_i)])

    return m, np.array(thetas), np.array(X)

class MDN_psi(nn.Module):

    def __init__(self, ndim_input=1, ndim_output=2, n_hidden=5, n_components=1):
        super(MDN_psi, self).__init__()
        self.fc_in = nn.Linear(ndim_input, n_hidden)
        self.tanh = nn.Tanh()
        self.m_out = nn.Linear(n_hidden, ndim_output)

    def forward(self, x):
        out = self.fc_in(x)
        act = self.tanh(out)
        out_m = self.m_out(act)
        return out_m

def train_psi(X, Y, model, optim, loss_fun, n_epochs=500, n_minibatch=50):
    dataset_train = [(x, y) for x, y in zip(X, Y)]

    losses = []

    for epoch in range(n_epochs):
        bgen = batch_generator(dataset_train, n_minibatch)

        for j, (x_batch, y_batch) in enumerate(bgen):
            x_var = Variable(torch.Tensor(x_batch))
            y_var = Variable(torch.LongTensor(y_batch)).view(n_minibatch)

            (out_act) = model(x_var)
            loss = lossfun(out_act, y_var)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.data[0])

        if (epoch + 1) % 100 == 0:
            print("[epoch %04d] loss: %.4f" % (epoch + 1, loss.data[0]))

    return model, optim, losses


# set prior parameters
shapes = [0.5, 7.5]
scales = [1.0, 1.0]

sample_size = 1
n_samples = 2000

n_epochs = 500
n_minibatch = 500

model = MDN_psi(n_hidden=10)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
lossfun = nn.CrossEntropyLoss()

m, theta, X = generate_dataset(n_samples, sample_size)
X, norm = normalize(X)

model_psi, optim_psi, losses = train_psi(X, m, model, optim, lossfun, n_epochs=n_epochs, n_minibatch=n_minibatch)

bf_true = []
bf_predicted = []
bf_stats =[]
model_indices = []
mi_pred = []

# gather summary stats
stats = []

softmax = nn.Softmax()

for i in range(100):

    # sample model index
    m_i = np.round(np.random.rand()).astype(int)

    # draw samples from the model given by the model index
    if m_i == 0:
        samples = np.random.poisson(0.5, sample_size)
    elif m_i == 1:
        samples = np.random.poisson(6.0, sample_size)

    # apply model for prediction
    stats_o = calculate_stats(samples).reshape(1, 1)
    stats_o, norm = normalize(stats_o, norm)

    stats.append(stats_o)

    X_var = Variable(torch.Tensor(stats_o))
    (out_act) = model(X_var)

    # in this vector, index 0 is Poi, index 1 is NB
    posterior_probs = softmax(out_act).data.numpy()[0]
    # predict the model with the larger posterior
    mi_pred.append(np.argmax(posterior_probs))

    # because we use a uniform prior the posterior ratio corresponds to the likelihood (evidence) ratio
    e0 = poisson_evidence(samples, shapes[0], scales[0], sample_size, log=True)
    e1 = poisson_evidence(samples, shapes[1], scales[1], sample_size, log=True)

    e0s = poisson_sum_evidence(samples, shapes[0], scales[0])
    e1s = poisson_sum_evidence(samples, shapes[1], scales[1])

    # calculate bf
    log_bftrue = e0 - e1
    log_bfstats = e0s - e1s
    log_bfpred = np.log(posterior_probs[0]) - np.log(posterior_probs[1])

    # append to lists
    bf_predicted.append(log_bfpred)
    bf_true.append(log_bftrue)
    bf_stats.append(log_bfstats)
    model_indices.append(m_i)

mi_true = np.array(model_indices)
# we predict m0 if logbf > 0
mi_ana = (np.array(bf_true) < 0.)
mi_pred = (np.array(bf_predicted) < 0.)
stats = np.array(stats).squeeze()

plt.figure(figsize=(15, 8))

plt.subplot(211)
plt.title('The summary stats of the two models are well separated')
plt.hist(stats[mi_true==0], label='model 0')
plt.hist(stats[mi_true==1], label='model 1')
plt.legend()

plt.subplot(212)
plt.title('Predicted and analytical log Bayes Factor')
plt.plot(bf_true, label=r'log $BF(x)$')
plt.plot(bf_stats, label=r'log $BF(s(x))$')
plt.plot(bf_predicted, '--', label='predicted log BF')
plt.ylabel('log Bayes factor')
plt.xlabel('different data sets')
plt.legend(loc=4)

plt.savefig('../figures/comparison_poisson_priors_N{}M{}.pdf'.format(n_samples, sample_size))
plt.show()