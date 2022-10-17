"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    n, d = X.shape                                              # (n,d)
    K, _ = mixture.mu.shape
    pj = mixture.p.reshape(-1,1)                                # (K,1)
    mu = mixture.mu                                             # (K,d)
    var = mixture.var.reshape(-1,1)                             # (K,1)

    ## Get normal distribution
    def multi_normal(mu, var, X):
        # mu -> (K,d)
        # var -> (K,1)
        # X -> (1,d)
        d = X.shape
        const = 1 / ((2 * np.pi * var) ** d / 2)                        # (K,1)
        exp_x = ((X - mu) ** 2)  # .reshape(-1,1)                       # (K,d)
        exp_x_j_assigned = np.max(exp_x,axis = 1).reshape(-1,1)         # (K,1)
        exp = np.exp(-1 / 2 * exp_x_j_assigned / var)                   # (K,1)
        return const * exp

    post = np.zeros((n,K))
    LL = 0
    for i in range(n): # for each user i

        # Normal gaussian
        Nu = multi_normal(mu,var,X[i,:])                                    # (K,1)
        # Posterior probability
        post_i= pj*Nu/(np.sum(pj*Nu).reshape(-1,1))                         # (K,1)/(1,1) = (K,1)
        post[i,:] = post_i.reshape(K,)
        # Log Likelihood
        LL_post = np.sum(post[i,:])
        LL += np.log(LL_post)


    return (post,LL)
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError
