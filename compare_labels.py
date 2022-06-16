import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import ternary
from ternary.helpers import project_sequence

from scipy.io import loadmat
from sklearn.metrics import fowlkes_mallows_score

# read the data from the control (fully autonomous) run
auto_order = np.loadtxt(fname='autonomous_predicted_labels.txt',
                        skiprows=1,
                        max_rows=1,
                        dtype=int)

auto_label = np.loadtxt(fname='autonomous_predicted_labels.txt',
                        skiprows=3)

# read the data from the run of interest
time_string = '2022-06-15-12-28'

test_order = np.loadtxt(fname=f'labels/predicted_labels_{time_string}.txt',
                        skiprows=1,
                        max_rows=1,
                        dtype=int)

test_label = np.loadtxt(fname=f'labels/predicted_labels_{time_string}.txt',
                        skiprows=3)

# import the experimental data, including...
data = loadmat('FeGaPd_full_data_220104a.mat')

# ...the "true" labels from the data set, and...
true_label = data['labels_col'][0][0].flatten()

# ...the composition data in cartesian coordinates
composition = data['C']
idx = [1, 2, 0]
cartesian = np.array(list(zip(*project_sequence(composition[:, idx]))))

# calculate the fowlkes mallows index of each run
num_experiments = len(test_label)

auto_scores = np.zeros(num_experiments)
test_scores = np.zeros(num_experiments)

for i in range(num_experiments):
    auto_scores[i] = fowlkes_mallows_score(true_label, auto_label[i])
    test_scores[i] = fowlkes_mallows_score(true_label, test_label[i])


def beautify(ax):
    '''
    Remove "chartjunk" from a matplotlib plot Axes object

    Parameters
    ----------
    ax: the Axes object to be "beautified"
    '''
    # remove the bounding box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # remove tick marks
    plt.minorticks_off()
    ax.tick_params('both', length=0)

    # enforce integer tick labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # remove box around the legend
    ax.legend(frameon=False)


# plot the fowlkes mallows index of each run
fig, ax = plt.subplots()
ax.plot(auto_scores, label='control')
ax.plot(test_scores, label='test')

ax.set_ybound([0, 1])

ax.set_xlabel('iteration #')
ax.set_ylabel('FMI')

beautify(ax)

# plot the order of samples of each run
fig, tax = ternary.figure(scale=0.6)

# first create a dictionary of all the indices and their locations
points = {}
for i in range(num_experiments):
    points[tuple(cartesian[auto_order[i], :])] = str(i+1)

for i in range(num_experiments):
    if test_order[i] != auto_order[i]:
        # the test made a different measurement that the control
        if tuple(cartesian[test_order[i], :]) in points:
            # the point was already measured by the control
            points[tuple(cartesian[test_order[i], :])] += f', {str(i+1)}'
        else:
            # the point hasn't been measured by the control
            points[tuple(cartesian[test_order[i], :])] = f"{str(i+1)}'"

tax.right_corner_label("$Fe_{0.4}Ga_{0.6}$", fontsize=14)
tax.top_corner_label("$Fe_{0.4}Pd_{0.6}$", fontsize=14)
tax.left_corner_label("$Fe$", fontsize=14)

tax.set_background_color('white')

tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()

tax.boundary()

for x, y in points:
    plt.text(x, y, points[(x, y)],
             fontsize=10,
             horizontalalignment='center',
             verticalalignment='center')

plt.show()
