# DPGMM
This repository is the source code of the **DPGMM**(Dirichlet Process Gaussian Mixture Model) for the clustering and variable selection of the functional linear regression data.

# How to use?
DPGMM offers user-friendly APIs similar to those in sklearn. Initially, we create an instance of the model class using specified parameters. Once instantiated, this model can then be employed for fitting and predicting data. There are three kinds of DPGMM model: `DPGMM_basic`, `DPGMM_mixed_effect`, `DPGMM_AR1`. `DPGMM_mixed_effect` assumes the mixed effect model, adding the random effect term to the fixed effect term that catches the group effect as well as global mean at the same time. `DPGMM_AR1` supposes the autoregressive model where the data at any point in time is a function of its previous value plus some random noise.

```python
from _DPGMM_basic import DPGMM_basic

model_configs = {'n_componetns':30, 'tol':1e-3, 'reg_covar':1e-6, 'max_iter':10000,
                  'random_state':42, 'verbose'=2, 'verbose_interval':10}
model = DPGMM_basic(**model_configs)
model.fit(X, y, counts)

model.weights_ # Weights of each cluster
model.beta_mean_ # Posterior mean of beta
np.sqrt(model.precision_rate_/model.precision_shape_) # Posterior mean of standard deviation
y_pred = model.predict(X, y, counts) # Predicted value for training data
```

# Parameters
- `n_components`: The number of mixture components. Default is 1.0
- `tol`: The convergence threshold. Default is 1e-3
- `reg_covar`: Non-negative regularization added to the diagonal of covariance. Default is 1e-6
- `max_iter`: The number of VI iterations to perform. Default is 200
- `Weight_concentration_prior`: The Dirichlet concentration of each component. Default is 1.0
- `beta_rho_prior`: The variance prior on the beta parameters that are not penalized. Default is 10**2
- `precision_shape_prior`: The prior of the shape parameter on the precision distribution. Default is 0.001
- `precision_rate_prior`: The prior of the rate parameter on the precision distribution. Default is 0.001      
- `shrink_shape_prior`: The prior of the shape parameter on the shrink distribution. Default is 0.001
- `shrink_rate_prior`: The prior of the rate parameter on the shrink distribution. Default is 0.001
- `random_state`: Controls the random seed given to the method chosen to initialize the parameters.  
- `verbose`: Enable verbose output. Default is 0
- `verbose_interval`: Number of iterations done before the next print. Default is 10
- `random_covariance_prior`: The prior of the covariance for random effect. Default is 50*np.eye(). (`DPGMM_mixed_effect` only)
- `random_degrees_prior`: The prior of the degress of freedom for random effect. Default is 1.0 (`DPGMM_mixed_effect` only)
- `corr_precision_prior`: The prior of the precision parameter of the correlation distributions. Dsfault is 0.001 (`DPGMM_AR1` only)

# Methods
- `fit(X, y, counts)`: Fit estimator.
- `predict(X, y, counts)`: Predict the labels of each subgroup for the data samples using the trained model

# Attributes
- `weights_`: The weights of each mixture component.
- `beta_mean_`: The transformed coefficients of each mixture component.
- `beta_star_mean_`: The updated coefficients of each mixture component.
- `beta_covariance_`: The updated covariance of each mixture component.
- `precision_shape_`: The updated shape parameter for each precision component in the mixture.
- `precision_rate_`: The updated rate parameter for each precision component in the mixture. 
- `converged_`: True when convergence was reached in fit(), False otherwise.
- `n_iter_`: Number of step used by the best fit of inference to reach the convergence.
- `weight_concentration_`: The Dirichlet concentration of each component.
- `wishart_degrees_`: The updated degrees of freedom parameter for each precision of random effect. (`DPGMM_mixed_effect` only)
- `wishart_covariance_`: The updated covariance parameter for each precision of random effect. (`DPGMM_mixed_effect` only)
- `mu_`: The updated mean parameter for each random effect component.  (`DPGMM_mixed_effect` only)
- `Sigma_`: The updated covariance parameter for each random effect component. (`DPGMM_mixed_effect` only)
- `W`: The design matrix for the random effect term by using the main design matrix X (`DPGMM_mixed_effect` only)
- `varpi_`: The updated mean parameter for each correlation component. (`DPGMM_AR1` only)
- `vartheta_`: The updated variance parameter for each correlation component. (`DPGMM_AR1` only)

# Examples for AR(1)
```python
import numpy as np
from ._DPGMM_basic import DPGMM_basic
from ._DPGMM_mixed import DPGMM_mixed
from ._DPGMM_AR1 import DPGMM_AR1, _expectation_corr
from ._example_generators import (
    data_generator_basic, data_generator_mixed_effect, data_generator_AR1, 
    graph_generator, graph_corr_check
)

# Generate Data
x, y, true_zeta, counts = data_generator_AR1(repetition=50, scale=0.1, 
                                             number_subgroups=500, random_seed=100)

# Generate design matrix using basis expansion
knot = np.linspace(0, 1, num=30, endpoint=False)
N = x.shape[0]
D = knot.shape[0]+4
B = np.zeros((N, D))
for i in range(N):
    B[i,:] = np.array([1, x[i], x[i]**2, x[i]**3] + [abs(x[i]-t)**3 for t in knot])

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
graph_generator(B, knot, counts, model.beta_star_mean_, model.beta_covariance_,
                model.precision_shape_, model.precision_rate_, predicted_label,
                percentage=0.95, graph_threshold = 100,
                option='line_without_minor', interval=True)
```

![result](https://github.com/user-attachments/assets/7bbacc47-3d13-4146-ab02-21153146835e)
![zeta1](https://github.com/user-attachments/assets/56ca5581-a2ea-430d-a782-131c8cb3c5ec)
![zeta2](https://github.com/user-attachments/assets/ef4039c9-2e46-4b40-8b81-37687414e453)
