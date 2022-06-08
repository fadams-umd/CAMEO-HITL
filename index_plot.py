import matplotlib.pyplot as plt
import numpy as np
import ternary
from ternary.helpers import project_sequence
from scipy.io import loadmat

data = loadmat('FeGaPd_full_data_220104a.mat')

# composition data in cartesian coordinates
composition = data['C']
idx = [1, 2, 0]
cartesian = np.array(list(zip(*project_sequence(composition[:, idx]))))

fig, tax = ternary.figure(scale=0.6)
fig.set_size_inches(12, 12)

tax.right_corner_label("$Fe_{0.4}Ga_{0.6}$", fontsize=20)
tax.top_corner_label("$Fe_{0.4}Pd_{0.6}$", fontsize=20)
tax.left_corner_label("$Fe$", fontsize=20)

tax.set_background_color('white')

tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()

tax.boundary()

for i in range(np.shape(cartesian)[0]):
    x, y = cartesian[i, :]
    plt.text(x, y, str(i), fontsize=8)

plt.show()