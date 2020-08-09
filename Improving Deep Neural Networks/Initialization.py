import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

# %matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()

print(train_X.shape, train_Y.shape)


print(int((1+2)/2))

# Kullback–Leibler divergence

import pandas as pd
import numpy as np

y = pd.Series([1,0])
yhat = pd.Series([0.1, 0.9])
def KL_divergence(Y, Yhat):
	if type(Y) == type(Yhat) == pd.core.series.Series and Y.shape==Yhat.shape:
		re = [Yhat[i] * np.log2(Yhat[i]) - Yhat[i] * np.log2(Y[i]) for i in range(len(Y))]

	return sum(re)


KL_divergence(y, yhat)

