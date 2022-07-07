import numpy as np
from scipy.io import loadmat
from ternary.helpers import project_sequence

import tensorflow as tf

from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import ternary

from sklearn.manifold import spectral_embedding
from sklearn.mixture import GaussianMixture

import gpflow

from scipy.stats import entropy

from sklearn.metrics import fowlkes_mallows_score

# read the data from the control (fully autonomous) run
auto_order = np.loadtxt(fname='autonomous_predicted_labels.txt',
                        skiprows=1,
                        max_rows=1,
                        dtype=int)

auto_label = np.loadtxt(fname='autonomous_predicted_labels.txt',
                        skiprows=3)

# import the experimental data, including the...
experimental_data = loadmat('FeGaPd_full_data_220104a.mat')

# ..."true" labels from the data set, ...
true_label = experimental_data['labels_col'][0][1].flatten()

# ...composition data in cartesian coordinates, ...
composition = experimental_data['C']
idx = [1, 2, 0]
cartesian = np.array(list(zip(*project_sequence(composition[:, idx]))))

# ...and x ray diffraction data
xrd = experimental_data['X'][:, 631:1181]

# seed the numpy and tensorflow random number generation for reproducability
seed = 2
rng = np.random.RandomState(seed)
tf.random.set_seed(seed)

# experiment iteration number
iteration = 10

measured = auto_order[:iteration - 1]

# calculate the similarity matrix
pairwise_distance = squareform(pdist(xrd[measured, :], 'cosine'))
similarity_matrix = np.exp(-(pairwise_distance**2) / (2 * 0.7**2))

# perform spectral clustering
num_clusters = 5

spectral_clusters = spectral_embedding(adjacency=similarity_matrix,
                                       n_components=num_clusters,
                                       random_state=rng)

mixture_model = GaussianMixture(n_components=num_clusters,
                                covariance_type='diag',
                                n_init=10,
                                random_state=rng).fit(spectral_clusters)

cluster_probabilities = mixture_model.predict_proba(spectral_clusters)

labels = np.argmax(cluster_probabilities, axis=1).flatten()

# calculate user labels
points = [(0.2, 0.3), (0.4, 0.3)]

x1, y1 = points[0]
x2, y2 = points[1]

# calculate the distance between each point the line
boundary_distance = [(x-x1)*(y2-y1)-(y-y1)*(x2-x1) for (x, y) in cartesian]

user_labels = [1 if d > 0 else 0 for d in boundary_distance]

# create Gaussian process classificiation model
user_kernel = gpflow.kernels.SquaredExponential(active_dims=[2],
                                                lengthscales=0.001,
                                                variance=0.1)
gpflow.utilities.set_trainable(user_kernel.lengthscales, False)
gpflow.utilities.set_trainable(user_kernel.variance, False)

composition_kernel = gpflow.kernels.Matern32(active_dims=[0, 1],
                                             lengthscales=[0.2, 0.2],
                                             variance=1)
gpflow.utilities.set_trainable(composition_kernel.variance, False)
gpflow.utilities.set_trainable(composition_kernel.lengthscales, False)

hitl_kernel = composition_kernel + user_kernel

invlink = gpflow.likelihoods.RobustMax(num_clusters)

gpflow.utilities.set_trainable(invlink.epsilon, True)

likelihood = gpflow.likelihoods.MultiClass(num_clusters, invlink=invlink)

# train the model on the data with user input
hitl_input = np.column_stack((cartesian, user_labels))

hitl_data = (hitl_input[measured, :], labels)

hitl_gp_model = gpflow.models.VGP(data=hitl_data,
                                  kernel=hitl_kernel,
                                  likelihood=likelihood,
                                  num_latent_gps=num_clusters)

opt = gpflow.optimizers.Scipy()
opt_result = opt.minimize(hitl_gp_model.training_loss,
                          hitl_gp_model.trainable_variables,
                          options={'maxiter': 1000})

# get the kernel variance parameters
base_composition_variance = hitl_gp_model.kernel.kernels[0].variance
base_user_input_variance = hitl_gp_model.kernel.kernels[1].variance


def get_labels(composition_factor, user_input_factor):
    new_composition_variance = base_composition_variance * composition_factor
    hitl_gp_model.kernel.kernels[0].variance.assign(new_composition_variance)

    new_user_input_variance = base_user_input_variance * user_input_factor
    hitl_gp_model.kernel.kernels[1].variance.assign(new_user_input_variance)

    y_mean, _ = hitl_gp_model.predict_y(hitl_input)

    return(np.argmax(y_mean.numpy(), axis=1).flatten())


def compare_labels(composition_factor, user_input_factor):
    control_fmi = fowlkes_mallows_score(
        true_label, auto_label[iteration - 1, :])

    test_label = get_labels(composition_factor, user_input_factor)
    test_fmi = fowlkes_mallows_score(true_label, test_label)

    return(test_fmi - control_fmi)


# print results to table
composition_factors = [0.5, 0.75, 1, 1.25, 1.5]
composition_vars = [base_composition_variance * f for f in composition_factors]

user_input_factors = [0.5, 0.75, 1, 1.25, 1.5]
user_input_vars = [base_user_input_variance * f for f in user_input_factors]
print('%10s %10f %10f %10f %10f %10f' % tuple([''] + user_input_vars))

for (cv, cf) in zip(composition_vars, composition_factors):
    results = [compare_labels(cf, uf) for uf in user_input_factors]

    print('%10f %10f %10f %10f %10f %10f' % tuple([cv] + results))
