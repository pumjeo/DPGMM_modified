"""Basic model for the DPGMM model"""

# Author: Neulpum Jeong <pumjeo@gmail.com>
# License: BSD 3 clause
# Time : 2024/08/31

import numpy as np
from numbers import Integral, Real
from scipy import linalg
from scipy.special import digamma, kv
from sklearn.utils._param_validation import Interval, StrOptions
from ._base import DPGMM_base, calculate_relative_change, check_convergence


def _estimate_gaussian_parameters(X, y, counts, resp, reg_covar):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.
        
    y : array-like of shape (n_samples,)
        response variables array
        
    counts : array-like of shape (n_subgroups, )
        The list of number of samples that belongs to each subgroup

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        
    nk2 : array-like of shape (n_components,)

    xk : array-like of shape (n_components, n_features)
    
    sk : array-like (n_components, n_features, n_features)
    
    tk : array-like of shape (n_components,)
    """
    _, n_features = X.shape
    _, n_components = resp.shape
    resp_duple = np.repeat(resp, counts, axis=0) 
    
    nk = np.dot(counts,resp) + 10 * np.finfo(resp.dtype).eps
    nk2 = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    
    xk = np.zeros((n_components, n_features))
    for k in range(n_components):
        xk[k] = np.sum(resp_duple[:,k][:,np.newaxis] * X * y[:,np.newaxis], axis=0)
    
    sk = np.zeros((n_components, n_features, n_features))
    for k in range(n_components): 
        sk[k] = np.dot(resp_duple[:, k] * X.T, X) 
        sk[k].flat[:: n_features + 1] += reg_covar 
    
    tk = np.dot(resp_duple.T, y**2)
    
    return nk, nk2, xk, sk, tk


def _expectation_shrink_inv(c_tau, d_tau, f_tau, rho):
    """Compute the expectation of covariance term of beta coefficients.

    Parameters
    ----------
    c_tau : float

    d_tau : array-like of shape (n_components,)
    
    f_tau : array-like of shape (n_components, n_features-4)
    
    rho : float
        The magnitude of the variance for the coefficients that are not penalized.
    
    Return
    ------
    tau_inv_expect : array-like of shape (n_components, n_features, n_features)
        The expectation of the covariance term of beta coefficients
    """
    n_components, n_features = f_tau.shape[0], f_tau.shape[1]+4
    
    numer = np.sqrt(d_tau)[:,np.newaxis] * kv(c_tau+1, np.sqrt(d_tau[:,np.newaxis]*f_tau)) # (C, D-4)
    denom = np.sqrt(f_tau) * kv(c_tau, np.sqrt(d_tau[:,np.newaxis]*f_tau)) # (C, D-4)
    temp = numer/denom - ((2*c_tau)/f_tau)

    tau_inv_expect = np.zeros((n_components, n_features, n_features)) # (C, D, D)
    for k in range(n_components):
        tau_inv_expect[k] = linalg.block_diag((1/rho)*np.eye(4), np.diag(temp[k])) 
    
    return tau_inv_expect


class DPGMM_basic(DPGMM_base):
    """Variational Bayesian estimation of a Gaussian mixture.

    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution. The effective number of
    components can be inferred from the data.

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components. Depending on the data and the value
        of the `weight_concentration_prior` the model can decide to not use
        all the components by setting some component `weights_` to values very
        close to zero. The number of effective components is therefore smaller
        than n_components.

    tol : float, default=1e-3
        The convergence threshold. VI iterations will stop when the
        relative change is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=200
        The number of VI iterations to perform.

    weight_concentration_prior : float or None, default=None
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). This is commonly called gamma in the
        literature. The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        mixture weights simplex. The value of the parameter must be greater
        than 0. If it is None, it's set to `1.0`.

    beta_rho_prior : float or None, default=None
        The variance prior on the beta parameters that are not penalized.
        Controls the extent of where the coefficients can be placed. Larger
        values allows the first four coefficients not to be shrinkaged.
        The value of the parameter must be greater than 0.
        If it is None, it is set to 10**10.

    precision_shape_prior : float or None, default=None
        The prior of the shape parameter on the precision
        distributions (Gamma). If it is None, it's set to `10**-10`.

    precision_rate_prior : float or array-like, default=None
        The prior of the rate parameter on the precision
        distributions (Gamma). If it is None, it's set to `10**-10`.
                
    shrink_shape_prior :
        The prior of the shape parameter on the shrinkage
        distributions (Gamma). If it is None, it's set to `0.001`.   
    
    shrink_rate_prior :
        The prior of the rate parameter on the shrinkage
        distributions (Gamma). If it is None, it's set to `0.001`.  

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    beta_mean_ : array-like of shape (n_components, n_features)
        The coefficients of each mixture component.
        It is transformed along with the original design matrix (X).

    beta_star_mean_ : array-like of shape (n_components, n_features)
        The coefficients of each mixture component. It is updated 
        parameters along with the standardized design matrix (X_star).   
    
    beta_covariance_ : array-like of shape (n_components, n_features, n_features)
        The covariance of each mixture component.

    precision_shape_ : array-like of shape (n_components,)
        The shape parameter for each precision component in the mixture. 

    precision_rate_ : array-like of shape (n_components,)
        The shape parameter for each precision component in the mixture.    
    
    c_tau_ : float
        The first parameter of the posterior distribution of shrinkage parameter (GIG)

    d_tau_ : array-like of shape (n_components,)
        The second parameter of the posterior distribution of shrinkage parameter (GIG)
        
    f_tau_ : array-like of shape (n_components, n_features-4)
        The third parameter of the posterior distribution of shrinkage parameter (GIG)
    
    shrink_shape_ : float
        The shape parameter of the posterior distribution of the manitude of shrinkage (Gamma)
    
    shrink_rate_ : array-like of shape (n_components,)
        The rate parameter of the posterior distribution of the manitude of shrinkage (Gamma)    
    
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of inference to reach the convergence.

    weight_concentration_prior_ : float
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the simplex.

    weight_concentration_ : array-like of shape (n_components,)
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet).

    beta_rho_prior_ : float
        The variance prior on the beta parameters that are not penalized.
        Controls the extent of where the coefficients can be placed. Larger
        values allows the first four coefficients not to be shrinkaged.
        The value of the parameter must be greater than 0.
        If it is None, it is set to 10**10.

    precision_shape_prior_ : float
        The prior of the shape parameter on the precision
        distributions (Gamma).

    precision_rate_prior_ : array-like of shape (n_components,)
        The prior of the rate parameter on the precision
        distributions (Gamma).
        
    shrink_shape_prior_ :
        The prior of the shape parameter on the shrinkage
        distributions (Gamma).    
        
    shrink_rate_prior_ : 
        The prior of the rate parameter on the shrinkage
        distributions (Gamma).

    """

    _parameter_constraints: dict = {
        **DPGMM_base._parameter_constraints,
        "weight_concentration_prior": [
            None,
            Interval(Real, 0.0, None, closed="neither"),
        ],
        "precision_shape_prior": [None, Interval(Real, 0.0, None, closed="neither")],
        "precision_rate_prior": [None, Interval(Real, 0.0, None, closed="neither")],
        "shrink_shape_prior": [None, Interval(Real, 0.0, None, closed="neither")],
        "shrink_rate_prior": [None, Interval(Real, 0.0, None, closed="neither")],
        "beta_rho_prior": [None, Interval(Real, 0.0, None, closed="neither")]
    }

    def __init__(
        self,
        *,
        n_components=1,
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=200,
        weight_concentration_prior=None,
        beta_rho_prior=None,
        precision_shape_prior=None,
        precision_rate_prior=None,
        shrink_shape_prior=None,
        shrink_rate_prior=None,
        random_state=None,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

        self.weight_concentration_prior = weight_concentration_prior
        self.beta_rho_prior = beta_rho_prior
        self.precision_shape_prior = precision_shape_prior
        self.precision_rate_prior = precision_rate_prior
        self.shrink_shape_prior = shrink_shape_prior
        self.shrink_rate_prior = shrink_rate_prior

    def _check_parameters(self, X):
        """Check that the parameters are well defined.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        
        """ 
        
        self._check_weights_parameters()
        self._check_betas_parameters(X)
        self._check_precisions_parameters()
        self._check_shrink_parameters()

    def _check_weights_parameters(self):
        """Check the parameter of the Dirichlet distribution."""

        if self.weight_concentration_prior is None:
            self.weight_concentration_prior_ = 1.0
        else:
            self.weight_concentration_prior_ = self.weight_concentration_prior  

    def _check_betas_parameters(self, X):
        """Check the parameters of the Gaussian distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        
        _, n_features = X.shape
        
        # beta_parameters
        if self.beta_rho_prior is None:
            self.beta_rho_prior_ = 10**10
        else:
            self.beta_rho_prior_ = self.beta_rho_prior
        
    def _check_precisions_parameters(self):
        """Check the prior parameters of the precision distribution."""

        if self.precision_shape_prior is None:
            self.precision_shape_prior_ = 10**-10
        elif self.precision_shape_prior > 0:
            self.precision_shape_prior_ = self.precision_shape_prior
        else:
            raise ValueError(
                "The parameter 'precision_shape_prior' "
                "should be greater than %d, but got %.3f."
                % (0, self.precision_shape_prior)
            )
            
        if self.precision_rate_prior is None:
            self.precision_rate_prior_ = 10**-10 
        elif self.precision_rate_prior > 0:
            self.precision_rate_prior_ = self.precision_rate_prior
        else:
            raise ValueError(
                "The parameter 'precision_rate_prior' "
                "should be greater than %d, but got %.3f."
                % (0, self.precision_rate_prior)
            )   

    def _check_shrink_parameters(self):
        """Check the prior parameters of the shrink distribution."""

        if self.shrink_shape_prior is None:
            self.shrink_shape_prior_ = 0.001 
        elif self.shrink_shape_prior > 0:
            self.shrink_shape_prior_ = self.shrink_shape_prior
        else:
            raise ValueError(
                "The parameter 'shrink_shape_prior' "
                "should be greater than %d, but got %.3f."
                % (0, self.shrink_shape_prior)
            )

        if self.shrink_rate_prior is None:
            self.shrink_rate_prior_ = 0.001 
        elif self.shrink_rate_prior > 0:
            self.shrink_rate_prior_ = self.shrink_rate_prior
        else:
            raise ValueError(
                "The parameter 'shrink_rate_prior' "
                "should be greater than %d, but got %.3f."
                % (0, self.shrink_rate_prior)
            )     

    def _initialize(self, X, y, counts, resp):
        """Initialization of the mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,)
        
        counts : array-like of shape (n_subgroups,)
            The list of number of samples that belongs to each subgroup
        
        resp : array-like of shape (n_subgroups, n_components)
        """
        
        _, n_features = X.shape 
        
        nk, nk2, xk, sk, tk = _estimate_gaussian_parameters(
            X, y, counts, resp, self.reg_covar
        )
        
        self.c_tau_ = 0.5 # Constant
        self.d_tau_ = np.repeat(1, self.n_components) # Random Initialization
        self.f_tau_ = np.array([np.repeat(1, n_features-4) for _ in range(self.n_components)]) # Random Initialization
        self.shrink_shape_ = self.shrink_shape_prior_ + n_features-4   # Constant
    
        self._estimate_weights(nk2)
        self._estimate_beta_star(xk, sk)
        self._estimate_precisions(nk, xk, tk)
        self._estimate_scale()   

    def _estimate_weights(self, nk2):
        """Estimate the parameters of the Dirichlet distribution.

        Parameters
        ----------
        nk2 : array-like of shape (n_components,)
        """

        self.weight_concentration_ = (
            1.0 + nk2,
            (
                self.weight_concentration_prior_
                + np.hstack((np.cumsum(nk2[::-1])[-2::-1], 0))
            ),
        )

    def _estimate_beta_star(self, xk, sk): 
        """Estimate the parameters of the Gaussian distribution.

        Parameters
        ----------
        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_components, n_features, n_features)
        """
        _, n_features = xk.shape
        
        tau_inv_expect = _expectation_shrink_inv(self.c_tau_, self.d_tau_, 
                                                    self.f_tau_, self.beta_rho_prior_)
        
        self.beta_covariance_ = np.linalg.inv(sk + tau_inv_expect)
        
        self.beta_star_mean_ = np.empty((self.n_components, n_features))    
        for k in range(self.n_components): 
            self.beta_star_mean_[k] = np.dot(self.beta_covariance_[k], xk[k])

    def _estimate_precisions(self, nk, xk, tk):
        """Estimate the precisions parameters of the precision distribution

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)
        
        tk : array-like of shape (n_components,)
        """
        _, n_features = xk.shape

        self.precision_shape_ = self.precision_shape_prior_ + 0.5 * nk
        
        self.precision_rate_ = np.empty((self.n_components))
        for k in range(self.n_components):
            self.precision_rate_[k] = (self.precision_rate_prior_ + 0.5 * tk[k] - 0.5 * xk[k].T
                                    @ self.beta_star_mean_[k]) 
        
    def _estimate_shrink(self):
        """Estimate the GIG parameters of the shrinkage distribution"""        
        
        self.d_tau_ = 2 * (self.shrink_shape_ / self.shrink_rate_)
        
        _, n_features = self.beta_star_mean_.shape
        self.f_tau_ = np.empty((self.n_components, n_features-4))
        for k in range(self.n_components):
            self.f_tau_[k] = ( 
                              (self.precision_shape_[k]/self.precision_rate_[k]) 
                              * (self.beta_star_mean_[k]**2) + np.diag(self.beta_covariance_[k])
                              )[4:]

    def _estimate_scale(self):
        """Estimate the rate parameter of the magnitude of shrinkage distribution"""
        
        numer = np.sqrt(self.f_tau_) * kv(self.c_tau_+1, np.sqrt(self.d_tau_[:,np.newaxis]*self.f_tau_))
        denom = np.sqrt(self.d_tau_)[:,np.newaxis] * kv(self.c_tau_, np.sqrt(self.d_tau_[:,np.newaxis]*self.f_tau_))
        self.shrink_rate_ = self.shrink_rate_prior_ + np.sum(numer/denom, axis=1)       

    def _m_step(self, X, y, counts, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,)
        
        counts : array-like of shape (n_subgroups,)
            The list of number of samples that belongs to each subgroup
        
        log_resp : array-like of shape (n_subgroups, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each subgroup in X.
        """

        nk, nk2, xk, sk, tk = _estimate_gaussian_parameters(
            X, y, counts, np.exp(log_resp), self.reg_covar
        )
        
        self._estimate_weights(nk2)
        self._estimate_beta_star(xk, sk)
        self._estimate_precisions(nk, xk, tk)
        self._estimate_scale() 
        self._estimate_shrink()        

    def _estimate_log_weights(self):
        digamma_sum = digamma(
            self.weight_concentration_[0] + self.weight_concentration_[1]
        )
        digamma_a = digamma(self.weight_concentration_[0])
        digamma_b = digamma(self.weight_concentration_[1])
        return (
            digamma_a
            - digamma_sum
            + np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1]))
        )     
        
    def _estimate_log_prob(self, X, y, counts):
        n_samples , n_features = X.shape     

        log_gauss = np.empty((n_samples, self.n_components))

        for k in range(self.n_components): 
            first = (self.precision_shape_[k]/self.precision_rate_[k]) * (y-np.dot(X, self.beta_star_mean_[k]))**2 
            second = np.einsum('ij,ij->i', X, np.dot(X, self.beta_covariance_[k])) 
            log_gauss[:,k] = -0.5 * (first+second) 

        log_lambda = 0.5 * (digamma(self.precision_shape_) - np.log(self.precision_rate_))
        temp = log_gauss + log_lambda
        temp_split = np.split(temp, np.cumsum(counts)[:-1], axis=0)
        res = [np.sum(ele, axis=0) for ele in temp_split]
        log_prob = np.vstack(res)
        return log_prob
    
    def _get_parameters(self):
        return (
            self.weight_concentration_,
            self.beta_star_mean_,
            self.beta_covariance_,
            self.precision_shape_,
            self.precision_rate_,
            self.d_tau_,
            self.f_tau_,
            self.shrink_rate_,
        )

    def _set_parameters(self, params):
        (
            self.weight_concentration_,
            self.beta_star_mean_,
            self.beta_covariance_,
            self.precision_shape_,
            self.precision_rate_,
            self.d_tau_,
            self.f_tau_,
            self.shrink_rate_,
        ) = params

        # Weights computation
        weight_dirichlet_sum = (
            self.weight_concentration_[0] + self.weight_concentration_[1]
        )
        tmp = self.weight_concentration_[1] / weight_dirichlet_sum
        self.weights_ = (
            self.weight_concentration_[0]
            / weight_dirichlet_sum
            * np.hstack((1, np.cumprod(tmp[:-1])))
        )
        self.weights_ /= np.sum(self.weights_)
