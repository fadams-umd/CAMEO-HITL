import numpy as np
import matplotlib.pyplot as plt
import gpflow
from sklearn.manifold import spectral_embedding
from sklearn.mixture import GaussianMixture
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist, squareform
from ternary.helpers import project_sequence


def phase_mapping(X, num_clusters):
    '''
    Cluster data using spectral clustering and a Gaussian mixture model

    Parameters
    ----------
    X: m x n matrix - m rows of n dimensional data
    num_clusters: number of groups to cluster data into

    Returns
    -------
    cl: clustering labels for each sample
    cluster_prob: the probability for each sample to belong to each cluster
    '''

    K = similarity_matrix(X, 'cosine')

    if X.shape[0] <= num_clusters:
        # fewer data points than clusters, each point gets its own cluster
        cluster_prob = np.eye(X.shape[0])
    else:
        x_se = spectral_embedding(adjacency=K, n_components=num_clusters)
        model = GaussianMixture(n_components=num_clusters,
                                covariance_type='diag',
                                n_init=100).fit(x_se)

        cluster_prob = model.predict_proba(x_se)

    cl = np.argmax(cluster_prob, axis=1).flatten()
    return cl, cluster_prob


def composition_to_graph(T):
    '''
    Use the composition data to identify nearby neighbors graph in
    composition-space.

    Parameters
    ----------
    T: n x 3 matrix - n rows of composition data in ternary coordinate space

    Returns
    -------
    S: n x n matrix - symmetric adjacency matrix
    '''

    N = T.shape[0]
    XYc = np.array(list(zip(*project_sequence(T))))
    vor = Voronoi(XYc)
    points_separated = vor.ridge_points
    S = np.zeros((N, N))
    for i in range(points_separated.shape[0]):
        S[points_separated[i, 0], points_separated[i, 1]] = 1
        S[points_separated[i, 1], points_separated[i, 0]] = 1
    return S


def plot_graph(S, XY):
    '''
    Plot the graph learned from composition positions.

    Parameters
    ----------
    S: n x n matrix - symmetric adjacency matrix
    XY: n x 2 matrix - n rows of composition data in cartesian coordinates
    '''

    r, c = np.nonzero(S)
    for i in range(r.shape[0]):
        xx = [XY[r[i], 0], XY[c[i], 0]]
        yy = [XY[r[i], 1], XY[c[i], 1]]
        plt.plot(xx, yy, c=np.asarray([.8, .8, .8]))
        plt.plot(xx, yy, 'k.')


def prune_graph(S, XY, dist_ratio, num_nearest_neighbors):
    '''
    Remove edges from the graph that connect points which are too far away.

    Parameters
    ----------
    S: n x n matrix - symmetric adjacency matrix
    XY: n x 2 matrix - n rows of composition data in cartesian coordinates
    dist_ratio: adjusts the cutoff distance
    num_nearest_neighbors: desired degree for data vertices
    '''

    D = squareform(pdist(XY))
    mD = np.sort(D, 0)
    mD = mD[num_nearest_neighbors, :]
    S_ = S.copy()
    for i in range(S.shape[0]):
        d = D[i, :]
        S_[i, d > dist_ratio*mD[i]] = 0
        S_[d > dist_ratio*mD[i], i] = 0
    return S_


def similarity_matrix(X, metric, sigma=1):
    '''
    Calculate and return the similarity matrix used in spectral clustering.

    Parameters
    ----------
    X: m x n matrix - m rows of n dimensional data
    metric: distance metric passed to scipy.spatial.distance.pdist()
    sigma: scaling factor for Gaussian radial basis function (default=1)
    '''

    distance = squareform(pdist(X, metric))
    W = np.exp(-(distance**2) / (2*sigma**2))
    return W


def gpc_phasemapping(xy_curr, labels_curr, xy_full, num_clusters,
                     weight_prior=None):
    '''
    Take clustering labels for the samples and then extrapolate them throughout
    composition space, segmenting the XY space into 'phase regions'.

    Parameters
    ----------
    xy_curr: cartesian coordinates of measured data points
    labels_curr: cluster labels for those data
    xy_full: cartesian coordinates of measured and new, query data points
    num_clusters: the number of clusters we're assuming exist
    weight_prior: variance coefficient of (optional) prior kernel

    Returns
    -------
    y_mean: data point label predictions
    y_variance: data point label variances
    f_mean: data point latent GP predictions
    f_variance: data point latent GP variances
    '''

    data = (xy_curr, labels_curr)

    if weight_prior is None:
        kernel = gpflow.kernels.Matern32(lengthscales=[1, 1])
    else:
        prior_kernel = gpflow.kernels.SquaredExponential(active_dims=[2],
                                                         lengthscales=0.001,
                                                         variance=weight_prior)

        # fix the prior kernel lengthscale and variance
        gpflow.utilities.set_trainable(prior_kernel.parameters[1], False)
        gpflow.utilities.set_trainable(prior_kernel.parameters[0], False)
                                                         
        kernel = gpflow.kernels.Matern32(active_dims=[0, 1]) + prior_kernel

    invlink = gpflow.likelihoods.RobustMax(num_clusters)

    likelihood = gpflow.likelihoods.MultiClass(num_clusters, invlink=invlink)

    model = gpflow.models.VGP(data=data,
                              kernel=kernel,
                              likelihood=likelihood,
                              num_latent_gps=num_clusters)

    # hyperparameter optimization
    opt = gpflow.optimizers.Scipy()
    _ = opt.minimize(model.training_loss,
                     model.trainable_variables,
                     options={'maxiter': gpflow.ci_utils.ci_niter(1000)})

    # Poisson process for the full XY coordinates
    y = model.predict_y(xy_full)
    y_mean = y[0].numpy()
    y_variance = y[1].numpy()

    # (non-squeezed) probabilistic function for class labels
    f = model.predict_f(xy_full)
    f_mean = f[0].numpy()
    f_variance = f[1].numpy()

    return y_mean, y_variance, f_mean, f_variance