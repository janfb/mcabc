import random
import torch
import tqdm

from torch.autograd import Variable


class Trainer:
    """
    Class for training an MDN using pytorch.
    """

    def __init__(self, model, optimizer=None, classification=False, verbose=False):
        """
        :param model: generative model object like defined in model_comparison.model.BaseModel
        :param optimizer: pytorch optimizer
        :param classification: flag for doing regression or classification. regression if False
        :param verbose:
        """
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
        """
        :param X: batch of data points
        :param Y: batch of target points
        :param n_epochs: training repetitions
        :param n_minibatch: minibatch size
        :return: a list of loss values over iterations.
        """
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


def batch_generator(dataset, batch_size=5):
    """
    Arange a given data set in batches of a given size
    :param dataset:
    :param batch_size:
    :return:
    """
    random.shuffle(dataset)
    n_full_batches = len(dataset) // batch_size
    for i in range(n_full_batches):
        idx_from = batch_size * i
        idx_to = batch_size * (i + 1)
        xs, ys = zip(*[(x, y) for x, y in dataset[idx_from:idx_to]])
        yield xs, ys
