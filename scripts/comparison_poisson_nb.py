import pickle
import time

from model_comparison.utils import *
from model_comparison.mdns import ClassificationSingleLayerMDN, Trainer
from scipy.stats import gamma, beta, nbinom, poisson


def do_poisson_nb_comparison(k1, k2, k3, theta2, theta3, seed, matched_means=False):
    # set params
    sample_size = 10
    n_samples = 100000
    n_epochs = 400
    n_minibatch = 1000
    n_test_samples = 200

    # set RNG
    np.random.seed(seed)

    time_stamp = time.strftime('%Y%m%d%H%M_')

    theta1 = (k2 * theta2 * k3 * theta3) / k1

    prior_lam = scipy.stats.gamma(a=k1, scale=theta1)
    prior_k = scipy.stats.gamma(a=k2, scale=theta2)
    prior_theta = scipy.stats.gamma(a=k3, scale=theta3)
    prior_r = prior_k
    # define prior p by change of variables
    def prior_p(p): np.power(1-p, -2) * prior_theta(p / (1 - p))

    # generate the all data with a given seed
    Xall, m_all = generate_poisson_nb_data_set(n_samples + n_test_samples, sample_size, prior_lam, prior_k, prior_theta,
                                        seed=seed,
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
    trainer = Trainer(model, optimizer, verbose=True, classification=True)

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

        nb_logevi = calculate_nb_evidence(x, k2, theta2, k3, theta3, log=True)
        poi_logevi = poisson_evidence(x, k=k1, theta=theta1, log=True)

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
                                   k1=k1, k2=k2, k3=k3, theta1=theta1, theta2=theta2, theta3=theta3))

    print('MSE: {}'.format(mse))

    filename = time_stamp + 'k1_{}_th1_{}_k2_{}_th2_{}_k3_{}_th3_{}_N{}M{}'.format(k1, theta1, k2, theta2, k3, theta3,
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
k2 = 5.
theta2 = 2.0

k3 = 2.
theta3s = np.logspace(-2, 1, 30)

# then the scale of the Gamma prior for the Poisson is given by
k1 = 2.0

mses = []

for theta3 in theta3s:
    mse = do_poisson_nb_comparison(k1, k2, k3, theta2, theta3, seed=seed, matched_means=matched_means)
    mses.append(mse)

print(theta3s, mses)
