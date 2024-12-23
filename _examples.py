"""Toy example for the DPGMM model"""

# Author: Neulpum Jeong <pumjeo@gmail.com>
# License: BSD 3 clause
# Time : 2024/09/05


import numpy as np
from ._DPGMM_basic import DPGMM_basic
from ._DPGMM_mixed import DPGMM_mixed
from ._DPGMM_AR1 import DPGMM_AR1, _expectation_corr
from ._example_generators import (
    data_generator_basic, data_generator_mixed_effect, data_generator_AR1, 
    graph_generator, graph_corr_check
)

"""Basic Model"""

# Generate data
x, y, counts  = data_generator_basic(poisson_parameter=10, scale=0.1, 
                                     number_subgroups=1000, random_seed=100)

x_normalized = (x - x.min()) / (x.max() - x.min()) # Normalize x

# Generate design matrix using basis expansion
knot = np.linspace(0, 1, num=32, endpoint=True)[1:-1]
N = x_normalized.shape[0]
D = knot.shape[0] + 4
B = np.zeros((N, D))
for i in range(N):
    B[i, :] = np.array([1, x_normalized[i], x_normalized[i]**2, x_normalized[i]**3] + 
                       [abs(x_normalized[i] - t)**3 for t in knot])

# Fit the model
model = DPGMM_basic(n_components=30, tol=1e-3, reg_covar = 1e-6, max_iter=10000, 
            random_state=42, verbose=2, verbose_interval=10).fit(B, y, counts)

# Check the results
model.weights_ # Weights of each cluster
model.beta_mean_ # Posterior mean of beta
np.sqrt(model.precision_rate_/model.precision_shape_) # Posterior mean of standard deviation
predicted_label = model.predict(B, y, counts) # Predicted label

# Draw the estimated graphs
graph_generator(B, x, knot, counts, model.beta_star_mean_, model.beta_covariance_, model.precision_shape_, 
                model.precision_rate_, predicted_label, percentage=0.95, 
                graph_threshold = 100, option='line_without_minor', interval=True)



"""Mixed Effect Model"""

# Generate data
x, y, xai, counts = data_generator_mixed_effect(repetition = 50, scale=0.1, number_subgroups=200, 
                                                random_seed=42, mode='basic')

x_normalized = (x - x.min()) / (x.max() - x.min()) # Normalize x

# Generate design matrix using basis expansion
knot = np.linspace(0, 1, num=32, endpoint=True)[1:-1]
N = x_normalized.shape[0]
D = knot.shape[0] + 4
B = np.zeros((N, D))
for i in range(N):
    B[i, :] = np.array([1, x_normalized[i], x_normalized[i]**2, x_normalized[i]**3] + 
                       [abs(x_normalized[i] - t)**3 for t in knot])

# Fit the model
model = DPGMM_mixed(n_components=30, n_features2=2, tol=1e-3, reg_covar = 1e-6, max_iter=10000, 
                    random_state=42, verbose=2, verbose_interval=50).fit(B, y, counts)

# Check the results
model.weights_ # Weights of each cluster
model.beta_mean_ # Posterior mean of beta
np.sqrt(model.precision_rate_/model.precision_shape_) # Posterior mean of standard deviation
predicted_label = model.predict(B, y, counts) # Predicted label

# Check the results - estimation of true xai
Sigma_diag = np.vstack([np.diag(mat) for mat in model.Sigma])
Sigma_sqrt_diag = np.sqrt(Sigma_diag)

upper = model.mu + 2 * Sigma_sqrt_diag
lower = model.mu - 2 * Sigma_sqrt_diag

# Check the results - estimation of true xai
Sigma_diag = np.vstack([np.diag(mat) for mat in model.Sigma])
Sigma_sqrt_diag = np.sqrt(Sigma_diag)

upper = model.mu + 2 * Sigma_sqrt_diag
lower = model.mu - 2 * Sigma_sqrt_diag

print("")
print("Total number of true Xai is : ", xai.shape[0] * xai.shape[1])
print("Total number of valid Xai is : ", np.sum((lower < xai) & (xai < upper)))

graph_generator(B, x, knot, counts, model.beta_star_mean_, model.beta_covariance_, model.precision_shape_, 
                model.precision_rate_, predicted_label, percentage=0.95, 
                graph_threshold = 1, option='line_without_minor', interval=False,
                simul=True, mode='basic')


"""AR1 Model"""

# Generate Data
x, y, true_zeta, counts = data_generator_AR1(repetition=50, scale=0.1, 
                                             number_subgroups=500, random_seed=100)

x_normalized = (x - x.min()) / (x.max() - x.min()) # Normalize x

# Generate design matrix using basis expansion
knot = np.linspace(0, 1, num=32, endpoint=True)[1:-1]
N = x_normalized.shape[0]
D = knot.shape[0] + 4
B = np.zeros((N, D))
for i in range(N):
    B[i, :] = np.array([1, x_normalized[i], x_normalized[i]**2, x_normalized[i]**3] + 
                       [abs(x_normalized[i] - t)**3 for t in knot])

# Fit the model
model = DPGMM_AR1(n_components=30, tol=1e-3, reg_covar = 1e-6, max_iter=10000, 
            random_state=42, verbose=2, verbose_interval=10).fit(B, y, counts)

# Check the result
model.weights_ # Weights of each cluster
model.beta_mean_ # Posterior mean of beta
np.sqrt(model.precision_rate_/model.precision_shape_) # Posterior mean of standard deviation
predicted_label = model.predict(B, y, counts) # Predicted label

# Check the results - estimation of true zeta
zeta_mean, zeta_squared_mean = _expectation_corr(model.varpi_, model.vartheta_)
zeta_std = np.sqrt(zeta_squared_mean - zeta_mean**2)

upper = zeta_mean + 2 * zeta_std
lower = zeta_mean - 2 * zeta_std

print("Total number of true Zeta is : ", true_zeta.shape[0])
print("Total number of valid Zeta is : ", np.sum((lower < true_zeta) & (true_zeta < upper)))
graph_corr_check(zeta_mean, zeta_std, true_zeta) # Draw the graph

# Draw the estimated graphs
graph_generator(B, x, knot, counts, model.beta_star_mean_, model.beta_covariance_, model.precision_shape_, 
                model.precision_rate_, predicted_label, percentage=0.95, 
                graph_threshold = 100, option='line_without_minor', interval=True)