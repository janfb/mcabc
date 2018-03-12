import pickle
import time
import tqdm

from model_comparison.utils import *
from model_comparison.mdns import ClassificationSingleLayerMDN, Trainer
from scipy.stats import gamma, beta, nbinom, poisson


def do_poisson_nb_comparison(shape_lambda, shape_k, shape_theta, scale_k, scale_theta, seed, matched_means=False,
                             save_data=False, verbose=False):
    # set params
    sample_size = 10
    n_samples = 100000
    n_epochs = 400
    n_minibatch = 1000
    n_test_samples = 200

    # set RNG
    np.random.seed(seed)

    time_stamp = time.strftime('%Y%m%d%H%M_')

    scale_lambda = (shape_k * scale_k * shape_theta * scale_theta) / shape_lambda

    prior_lam = scipy.stats.gamma(a=shape_lambda, scale=scale_lambda)
    prior_k = scipy.stats.gamma(a=shape_k, scale=scale_k)
    prior_theta = scipy.stats.gamma(a=shape_theta, scale=scale_theta)
    prior_r = prior_k
    # define prior p by change of variables
    def prior_p(p): np.power(1-p, -2) * prior_theta(p / (1 - p))

    # generate the all data with a given seed
    Xall, m_all = generate_poisson_nb_data_set(n_samples + n_test_samples, sample_size, prior_lam, prior_k, prior_theta,
                                        matched_means=matched_means)

    # separate training and testing data
    X = Xall[:n_samples, :]
    m = m_all[:n_samples]
    Xtest = Xall[n_samples:, :]
    mtest = m_all[n_samples:]

    SX = calculate_stats_toy_examples(X)
    SX, training_norm = normalize(SX)

    # define the nn and trainer
    model = ClassificationSingleLayerMDN(ndim_input=2, n_hidden=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, optimizer, verbose=verbose, classification=True)

    # train with training data
    loss_trace = trainer.train(SX, m, n_epochs=n_epochs, n_minibatch=n_minibatch)

    # generate test data
    SXtest = calculate_stats_toy_examples(Xtest)
    SXtest, training_norm = normalize(SXtest, training_norm)

    # predict test data
    ppoi_predicted = []
    softmax = nn.Softmax(dim=0)

    for xtest in SXtest:
        # get activation
        out_act = model(Variable(torch.Tensor(xtest)))
        # normalize
        p_vec = softmax(out_act).data.numpy()
        # in this vector, index 0 is Poi, index 1 is NB
        ppoi = p_vec[0]
        ppoi_predicted.append(ppoi)

    ppoi_predicted = np.array(ppoi_predicted)
    # avoid overflow in log
    ppoi_predicted[ppoi_predicted == 1.] = 0.99999
    ppoi_predicted[ppoi_predicted < 1e-12] = 1e-12
    # calculate log bayes factor: equivalent to log ratio of posterior probs
    logbf_predicted = np.log(ppoi_predicted) - np.log(1 - ppoi_predicted)

    # calculate analytical log bayes factors
    nb_logevidences = []
    poi_logevidences = []

    for ii, (x, mi) in enumerate(zip(Xtest, mtest)):

        nb_logevi = calculate_nb_evidence(x, shape_k, scale_k, shape_theta, scale_theta, log=True)
        poi_logevi = poisson_evidence(x, k=shape_lambda, theta=scale_lambda, log=True)

        nb_logevidences.append(nb_logevi)
        poi_logevidences.append(poi_logevi)

    # calculate posterior probs from log evidences
    poi_logevidences = np.array(poi_logevidences)
    nb_logevidences = np.array(nb_logevidences)

    # just subtract to get the ratio of evidences
    logbf_ana = poi_logevidences - nb_logevidences

    # the analytical poisson posterior is just the poisson evidence normalized with the sum of all evidences
    ppoi_ana = calculate_pprob_from_evidences(np.exp(poi_logevidences), np.exp(nb_logevidences))
    mse = calculate_mse(ppoi_predicted, ppoi_ana)

    if save_data:
        # set up the result dict
        result_dict = dict(priors=dict(prior_k=prior_k, prior_theta=prior_theta, prior_lam=prior_lam),
                           data=dict(X=X, SX=SX, norm=training_norm, Xtest=Xtest, SXtest=SXtest, m=m, mtest=mtest),
                           model=dict(model=model, trainer=trainer, loss_trace=loss_trace),
                           results=dict(poi_logevidences=poi_logevidences,
                                        nb_logevidences=nb_logevidences,
                                        logbf_predicted=logbf_predicted,
                                        logbf_ana=logbf_ana,
                                        ppoi_predicted=ppoi_predicted,
                                        ppoi_ana=ppoi_ana,
                                        mse=mse),
                           params=dict(n_samples=n_samples, sample_size=sample_size, n_test_samples=n_test_samples,
                                       n_epochs=n_epochs, n_minibatch=n_minibatch, seed=seed,
                                       k1=shape_lambda, k2=shape_k, k3=shape_theta, theta1=scale_lambda, theta2=scale_k, theta3=scale_theta))

        print('MSE: {}'.format(mse))

        filename = time_stamp + 'k1_{}_th1_{}_k2_{}_th2_{}_k3_{}_th3_{}_N{}M{}'.format(shape_lambda, scale_lambda, shape_k, scale_k, shape_theta, scale_theta,
                                                                                       n_samples, sample_size)
        if matched_means:
            filename += '_mm'

        folder = '../data'
        full_path_to_file = os.path.join(folder, filename + '.p')
        with open(full_path_to_file, 'wb') as outfile:
            pickle.dump(result_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    return mse


# set seed
seed = 7
matched_means = False

# set priors
shape_k = 5.
scale_k = 2.0

shape_thetas = np.linspace(2, 10, 10)
scale_thetas = np.logspace(-2, 1, 30)

# then the scale of the Gamma prior for the Poisson is given by
shape_lambda = 2.0

mses = np.zeros((shape_thetas.shape[0], scale_thetas.shape[0]))

# progress bar
maxiter = mses.size

with tqdm.tqdm(total=maxiter) as pbar:

    for i, shape_th in enumerate(shape_thetas):
        for j, scale_th in enumerate(scale_thetas):
            mse = do_poisson_nb_comparison(shape_lambda, shape_k, shape_th, scale_k, scale_th, seed=seed,
                                           matched_means=matched_means)
            mses[i, j] = mse
            pbar.update(1)

# saving
time_stamp = time.strftime('%Y%m%d%H%M_')
filename = time_stamp + 'k1_{}_k2_{}_th2_{}_k3_{}_th3_{}'.format(shape_lambda, shape_k, scale_k,
                                                                 shape_thetas.shape[0],
                                                                 scale_thetas.shape[0])
if matched_means:
    filename += '_mm'

result_dict = dict(mses=mses,
                   shape_lambda=shape_lambda,
                   shape_k=shape_k,
                   scale_k=scale_k,
                   shape_thetas=shape_thetas,
                   scale_thetas=scale_thetas,
                   matched_means=matched_means,
                   seed=seed)

folder = '../data'
full_path_to_file = os.path.join(folder, filename + '.p')
with open(full_path_to_file, 'wb') as outfile:
    pickle.dump(result_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

print('Done')
