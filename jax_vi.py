import jax.numpy as jnp
from jax import random, grad, jit, jacfwd, lax, vmap, jacrev
from jax.scipy.linalg import inv, det
from jax.lax import scan
import jax
from jax import jit
from scipy.linalg import solve_discrete_are
from tqdm import tqdm

@jit
def KL_gaussian(m1, C1, m2, C2):
    """
    Computes the Kullback-Leibler divergence between two Gaussian distributions.
    m1, C1: Mean and covariance of the first Gaussian distribution.
    m2, C2: Mean and covariance of the second Gaussian distribution.
    """
    C2_inv = inv(C2)
    log_det_ratio = (jnp.log(jnp.linalg.eigvals(C2)).sum() - jnp.log(jnp.linalg.eigvals(C1)).sum()).real # log(det(C2) / det(C1)), works better with limited precision because the determinant is practically 0
    return 0.5 * (log_det_ratio - n + jnp.trace(C2_inv @ C1) + ((m2 - m1).T @ C2_inv @ (m2 - m1)))

@jit
def log_likelihood(v, y):
    """
    v: State estimates.
    y: Observations.
    """
    def log_likelihood_j(_, v_y):
        v_j, y_j = v_y
        error = y_j - H @ v_j
        ll = error.T @ inv_R @ error
        return _, ll
    _, lls = scan(log_likelihood_j, None, (v[1:, :], y))
    sum_ll = sum(lls)
    return -0.5 * sum_ll - 0.5 * (J - J0) * jnp.log(2 * jnp.pi) - 0.5 * (J - J0) * jnp.log(det(R))

@jit
def KL_sum(m, C, K, key):
    """
    Computes the sum of KL divergences between the predicted and updated state distributions.
    m: Estimated states.
    C: Covariance matrices of the state estimates.
    K: Kalman gain matrix.
    N: Number of samples to average over for Monte Carlo approximation.
    """
    def KL_j(_, m_C_y):
        m_prev, m_curr, C_prev, C_curr, key = m_C_y
        key, *subkeys_inner = random.split(key, num=N)
        def inner_map(subkey):
            perturbed_state = m_prev + random.multivariate_normal(subkey, jnp.zeros(n), C_prev)
            v_pred = state_transition_function(perturbed_state, dt, F)
            return KL_gaussian(m_curr, C_curr, v_pred, Q)

        mean_kl = jnp.mean(jax.lax.map(inner_map, jnp.vstack(subkeys_inner)), axis=0)
        return _, mean_kl

    _, *subkeys = random.split(key, num=J+1)
    _, mean_kls = scan(KL_j, None, (m[:-1, :], m[1:, :], C[:-1, :, :], C[1:, :, :], jnp.vstack(subkeys)))

    kl_sum = sum(mean_kls)

    return kl_sum

@jit
def var_cost(K, key):
    """
    Computes the cost function for optimization, combining KL divergence and log-likelihood.
    K: Kalman gain matrix.
    N: Number of samples for Monte Carlo approximation in KL divergence.
    """
    m, C = filtered(K)
    #print(jax.device_get(m), jax.device_get(C))

    key, *subkeys = random.split(key, num=N+1)
    def inner_map(subkey):
        return log_likelihood(random.multivariate_normal(subkey, m, C), y)
    #print(KL_sum(m, C,K,key))
    return (KL_sum(m, C, K, key) - jnp.mean(jax.lax.map(inner_map, jnp.vstack(subkeys))))  