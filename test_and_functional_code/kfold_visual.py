from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import tensorflow as tf
import scipy.io
import sklearn
from sklearn.model_selection import train_test_split


TRAIN_SET_PERCENTILES = 0.7
VALI_SET_PERCENTILES = 0.15
TEST_SET_PERCENTILES = 0.15
K_FOLD_SPLITS = 10


# Matplotlib color set
np.random.seed(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = K_FOLD_SPLITS

# Generate the class/group data
X = np.random.randn(100, 10)

percentiles_classes = [TRAIN_SET_PERCENTILES, VALI_SET_PERCENTILES, TEST_SET_PERCENTILES]
y = np.hstack([[ii] * int(100 * perc) for ii, perc in enumerate(percentiles_classes)])

# Evenly spaced cv groups repeated once
k_fold_groups = np.hstack([[ii] * 10 for ii in range(K_FOLD_SPLITS)])


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, 100],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax


fig, ax = plt.subplots()
cv = KFold(n_splits)
plot_cv_indices(cv, X, y, k_fold_groups, ax, n_splits)
ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
# Make the legend fit
plt.tight_layout()
plt.show()
