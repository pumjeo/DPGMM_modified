"""Base class for mixture model"""

# Author: Neulpum Jeong <pumjeo@gmail.com>
# License: BSD 3 clause
# Time : 2024/08/31

import numpy as np
from abc import ABC, abstractmethod
from numbers import Integral, Real
from time import time

from scipy import linalg
from scipy.special import digamma, kv, logsumexp
from scipy.stats import norm
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state


def calculate_relative_change(curr, prev, tol):
    """
    Calculates the relative change between current and previous parameters.

    Parameters:
    - curr (np.ndarray or tuple): Current variational parameters.
    - prev (np.ndarray or tuple): Previous variational parameters.
    - tol (float): Threshold for relative change to consider convergence.

    Returns:
    - float: The calculated relative change.
    """
    if isinstance(curr, tuple) and isinstance(prev, tuple):
        relative_changes = [calculate_relative_change(c, p, tol) for c, p in zip(curr, prev)]
        return max(relative_changes)  # Use max to ensure conservative convergence check
    else:
        diff_norm = np.linalg.norm(curr - prev)
        prev_norm = np.linalg.norm(prev)
        return diff_norm / prev_norm if prev_norm != 0 else np.inf

def check_convergence(current_params, previous_params, tol):
    """
    Checks the relative change between current and previous variational parameters for all parameters.

    Parameters:
    - current_params (list of np.ndarray or tuple): List of current variational parameters of different dimensions.
    - previous_params (list of np.ndarray or tuple): List of previous variational parameters of different dimensions.
    - tol (float): Threshold for relative change to consider convergence.

    Returns:
    - bool: True if relative change for all parameters is less than epsilon, indicating convergence.
    - list of float: The calculated relative changes for each parameter.
    """
    relative_changes = []
    for curr, prev in zip(current_params, previous_params):
        relative_change = calculate_relative_change(curr, prev, tol)
        relative_changes.append(relative_change)
        
    formatted_changes = [f"{change:.5f}" for change in relative_changes]
    
    has_converged = all(change < tol for change in relative_changes)
    return has_converged, formatted_changes


class DPGMM_base(BaseEstimator, ABC):
    """Base class for DPGMM models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "reg_covar": [Interval(Real, 0.0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
        "verbose_interval": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_components,
        tol,
        reg_covar,
        max_iter,
        random_state,
        verbose,
        verbose_interval,
    ):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        
    @abstractmethod
    def _check_parameters(self, X):
        """Check initial parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        """
        pass

    def _initialize_parameters(self, X, y, counts, random_state):
        """Initialize the model parameters randomly.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        
        y : array-like of shape (n_sample,)
        
        counts : array-like of shape (n_subgroups, )
            The list of number of samples that belongs to each subgroup
        
        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_subgroups = counts.shape[0]

        resp = random_state.uniform(size=(n_subgroups, self.n_components))
        resp /= resp.sum(axis=1)[:, np.newaxis]
          
        self._initialize(X, y, counts, resp)
        
        return np.log(resp)

    @abstractmethod
    def _initialize(self, X, y, counts, resp):
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        
        y : array-like of shape (n_sample,)
        
        counts : array-like of shape (n_subgroups, )

        resp : array-like of shape (n_samples, n_components)
        """
        pass

    def fit(self, X, y, counts):
        """Estimate model parameters with coordinate ascent variational inference.

        the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional covariates points. Each row
            corresponds to a single data point.

        y : array-like of shape (n_sample,)
            List of response variables
            
        counts : array-like of shape (n_subgroups, )
            The list of number of samples that belongs to each subgroup
            
        Returns
        -------
        self : object
            The fitted mixture.
        """
        # parameters are validated in fit_predict
        self.fit_predict(X, y, counts)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict(self, X, y, counts):
        """Estimate model parameters using observations and predict the labels.

        the method iterates between E-step and M-step for `max_iter`
        times until the relative change of each variational parameter is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like of shape (n_sample,)
            List of response variables
            
        counts : array-like of shape (n_subgroups,)
            The list of number of samples that belongs to each subgroup

        Returns
        -------
        labels : array, shape (n_subgroups,)
            Component labels.
        """
        X, y = self._validate_data(X, y, dtype=[np.float64, np.float32], ensure_min_samples=2) ###
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)
        
        # Design Matrix Standardization
        intercept = X[:, 0]
        mean_X = np.mean(X[:, 1:], axis=0) 
        std_dev_X = np.std(X[:, 1:], axis=0)
        standardized = (X[:, 1:] - mean_X) / std_dev_X
        X_star = np.column_stack((intercept, standardized))

        self.converged_ = False

        random_state = check_random_state(self.random_state)

        self._print_verbose_msg_init_beg()


        log_resp = self._initialize_parameters(X_star, y, counts, random_state)

        if self.max_iter == 0:
            best_params = self._get_parameters()
            best_n_iter = 0
        else:
            for n_iter in range(1, self.max_iter + 1):

                prev_params = self._get_parameters() + (log_resp,)
                
                _ , log_resp = self._e_step(X_star, y, counts) 
                self._m_step(X_star, y, counts, log_resp)
                
                curr_params = self._get_parameters() + (log_resp,)
                indicator, rel_change = check_convergence(curr_params, prev_params, self.tol)
        
                self._print_verbose_msg_iter_end(n_iter, rel_change)

                if indicator:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end()

            best_params = self._get_parameters()
            best_n_iter = n_iter

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                "The model did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data.",
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X, y) are always consistent with fit(X, y).predict(X, y)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X_star, y, counts)
        
        # Transformation for betas
        _, n_features = X_star.shape
        
        self.beta_mean_ = np.zeros((self.n_components, n_features))
        for k in range(self.n_components):
            mean_coef = self.beta_star_mean_[k][1:]/std_dev_X
            mean_const = np.array([self.beta_star_mean_[k][0]-np.sum(mean_X*mean_coef)])
            self.beta_mean_[k] = np.concatenate((mean_const, mean_coef), axis=0)
            #self.beta_star_mean_[k] = np.concatenate((mean_const, self.beta_star_mean_[k][1:]), axis=0)        
        
        return log_resp.argmax(axis=1)

    def _e_step(self, X, y, counts):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_sample,)
        
        counts : array-like of shape (n_subgroups,)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each subgroup in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each subgroup in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, y, counts)
        return np.mean(log_prob_norm), log_resp

    @abstractmethod
    def _m_step(self, X, y, counts, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_sample,)
        
        counts : array-like of shape (n_subgroups,)
            The list of number of samples that belongs to each subgroup

        log_resp : array-like of shape (n_subgroups, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each subgroup in X.
        """
        pass

    @abstractmethod
    def _get_parameters(self):
        pass

    @abstractmethod
    def _set_parameters(self, params):
        pass
    
    def predict(self, X, y, counts):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like of shape (n_sample,)
            List of response variables
            
        counts : array-like of shape (n_subgroups,)
            The list of number of samples that belongs to each subgroup

        Returns
        -------
        labels : array, shape (n_subgroups,)
            Component labels.
        """
        check_is_fitted(self)
        X, y = self._validate_data(X, y, reset=False)
        
        # Design Matrix Standardization
        intercept = X[:, 0]
        mean_X = np.mean(X[:, 1:], axis=0) 
        std_dev_X = np.std(X[:, 1:], axis=0)
        standardized = (X[:, 1:] - mean_X) / std_dev_X
        X_star = np.column_stack((intercept, standardized))
        
        return self._estimate_weighted_log_prob(X_star, y, counts).argmax(axis=1)

    def predict_proba(self, X, y, counts):
        """Evaluate the components' density for each subgroup.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like of shape (n_sample,)
            List of response variables
            
        counts : array-like of shape (n_subgroups,)
            The list of number of samples that belongs to each subgroup

        Returns
        -------
        resp : array, shape (n_subgroups, n_components)
            Density of each Gaussian component for each subgroup in X.
        """
        check_is_fitted(self)
        X, y = self._validate_data(X, y, reset=False)
        
        # Design Matrix Standardization
        intercept = X[:, 0]
        mean_X = np.mean(X[:, 1:], axis=0) 
        std_dev_X = np.std(X[:, 1:], axis=0)
        standardized = (X[:, 1:] - mean_X) / std_dev_X
        X_star = np.column_stack((intercept, standardized))
        
        _, log_resp = self._estimate_log_prob_resp(X_star, y, counts)
        return np.exp(log_resp)
        
    def _estimate_weighted_log_prob(self, X, y, counts):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_sample,)
        
        counts : array-like of shape (n_subgroups,)
            The list of number of samples that belongs to each subgroup

        Returns
        -------
        weighted_log_prob : array, shape (n_subgroup, n_component)
        """
        return self._estimate_log_prob(X, y, counts) + self._estimate_log_weights()

    @abstractmethod
    def _estimate_log_weights(self):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        pass

    @abstractmethod
    def _estimate_log_prob(self, X, y, counts):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each subgroup.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_sample,)
        
        counts : array-like of shape (n_subgroups,)

        Returns
        -------
        log_prob : array, shape (n_subgroups, n_component)
        """
        pass

    def _estimate_log_prob_resp(self, X, y, counts):
        """Estimate log probabilities and responsibilities for each subgroup.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each subgroup in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_sample,)
        
        counts : array-like of shape (n_subgroups,)
            The list of number of samples that belongs to each subgroup

        Returns
        -------
        log_prob_norm : array, shape (n_subgroups,)
            log p(X)

        log_responsibilities : array, shape (n_subgroups, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X, y, counts)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp    
    
    def _print_verbose_msg_init_beg(self):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization Completed")
        elif self.verbose >= 2:
            print("Initialization Completed")
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print(
                    "Iteration %d\t time lapse %.5fs\t"
                    % (n_iter, cur_time - self._iter_prev_time)
                )
                print("Rel change : ", diff_ll)
                self._iter_prev_time = cur_time

    def _print_verbose_msg_init_end(self):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose >= 2:
            print(
                "Initialization converged: %s\t time lapse %.5fs\t"
                % (self.converged_, time() - self._init_prev_time)
            )