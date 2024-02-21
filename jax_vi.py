import jax.numpy as np
from jax import random, grad, jit, jacfwd, lax, vmap, jacrev
from jax.scipy.linalg import inv, det
from jax.lax import scan
import jax
from jax import jit
from scipy.linalg import solve_discrete_are
from tqdm import tqdm
"""
Variable Descriptions:

Kalman Filter Parameters:
- K: Kalman gain matrix.

State Transition Parameters:
- n: Number of state variables.
- state_transition_function: Function to compute state transitions, encapsulating dynamics.
- Q: Process noise covariance matrix.
- jacobian: Function to compute the Jacobian matrix of the state transition function.

Observation Model Parameters:
- H: Observation matrix.
- R: Observation noise covariance matrix.
- inv_R: Inverse of the observation noise covariance matrix (if directly used).

Initial State Parameters:
- m0, C0: Initial state estimate and covariance matrix.

Estimation and Observation Data:
- v: State estimates.
- y: Observations.
- m1, C1: Mean and covariance matrix of the first Gaussian distribution.
- m2, C2: Mean and covariance matrix of the second Gaussian distribution.

Algorithm Parameters:
- J: Total number of observations.
- J0: Offset for the observation count, used in log-likelihood calculation.
- key: PRNG key for random number generation.
- N: Number of samples for Monte Carlo approximation.

Functionality:
- filtered_func: Higher-order function to apply filtering based on provided `filter_step_func`.
- filter_step_func: Function to perform a single forecast and update step in filtering.

"""


@jit
def KL_gaussian(n, m1, C1, m2, C2):
    """
    Computes the Kullback-Leibler divergence between two Gaussian distributions.
    m1, C1: Mean and covariance of the first Gaussian distribution.
    m2, C2: Mean and covariance of the second Gaussian distribution.
    n: number of state variables
    """
    C2_inv = inv(C2)
    log_det_ratio = (np.log(np.linalg.eigvals(C2)).sum() - np.log(np.linalg.eigvals(C1)).sum()).real # log(det(C2) / det(C1)), works better with limited precision because the determinant is practically 0
    return 0.5 * (log_det_ratio - n + np.trace(C2_inv @ C1) + ((m2 - m1).T @ C2_inv @ (m2 - m1)))

@jit
def log_likelihood(v, y, H, inv_R, R, J, J0):
    """
    Computes the log-likelihood of observations given state estimates.
    """
    def log_likelihood_j(_, v_y):
        v_j, y_j = v_y
        error = y_j - H @ v_j
        ll = error.T @ inv_R @ error
        return _, ll
    _, lls = scan(log_likelihood_j, None, (v[1:,:], y))
    sum_ll = sum(lls)
    return -0.5 * sum_ll - 0.5 * (J - J0) * np.log(2 * np.pi) - 0.5 * (J - J0) * np.log(det(R))

@jit
def KL_sum(m, C, K, n, state_transition_function, Q, key, N):
    """
    Computes the sum of KL divergences between the predicted and updated state distributions.
    """
    def KL_j(_, m_C_y):
        m_prev, m_curr, C_prev, C_curr, key = m_C_y
        key, *subkeys_inner = random.split(key, num=N)
        def inner_map(subkey):
            perturbed_state = m_prev + random.multivariate_normal(subkey, np.zeros(n), C_prev)
            v_pred = state_transition_function(perturbed_state)
            return KL_gaussian(m_curr, C_curr, v_pred, Q)
        mean_kl = np.mean(np.lax.map(inner_map, np.array(subkeys_inner)), axis=0)
        return _, mean_kl
    _, mean_kls = scan(KL_j, None, (m[:-1, :], m[1:, :], C[:-1, :, :], C[1:, :, :], np.array(random.split(key, num=m.shape[0]-1))))
    kl_sum = sum(mean_kls)
    return kl_sum

@jit
def var_cost(K, m0, C0, n, state_transition_function, Q, jacobian, H, R, y, key, N, J, J0, filtered_func, filter_step_func):
    """
    Computes the cost function for optimization, combining KL divergence and log-likelihood.
        J, J0, H, inv_R, R, n: Parameters for log_likelihood calculation.
    """
    m, C = filtered_func(K, y, m0, C0, n, state_transition_function, Q, jacobian, filter_step_func)
    key, *subkeys = random.split(key, num=N+1)
    log_likelihood_vals = np.lax.map(lambda subkey: log_likelihood(random.multivariate_normal(subkey, m, C), y, H, inv(R), R, J, J0), np.array(subkeys))
    return (KL_sum(m, C, K, n, state_transition_function, Q, key, N) - np.mean(log_likelihood_vals))

@jit
def filter_step(m_C_prev, y_curr, K, n, state_transition_function, Q, jacobian, H, R):
    """
    Apply a single forecast and update step using the Kalman filter.
    """
    m_prev, C_prev = m_C_prev
    m_pred = state_transition_function(m_prev)
    F_jac = jacobian(m_prev)
    m_update = (np.eye(n) - K @ H) @ m_pred + K @ y_curr
    C_pred = F_jac @ C_prev @ F_jac.T + Q
    C_update = (np.eye(n) - K @ H) @ C_pred @ (np.eye(n) - K @ H).T + K @ R @ K.T
    return (m_update, C_update), (m_update, C_update)

@jit
def filtered(K, m0, C0, n, state_transition_function, Q, jacobian, H, R, y, filter_step_func):
    """
    Applies the filtering process to estimate the system state over time.
    """
    _, m_C = scan(lambda m_C_prev, y_curr: filter_step_func(m_C_prev, y_curr, K, n, state_transition_function, Q, jacobian, H, R))

    m, C = m_C
    return np.vstack((m0, m)), np.vstack((C0.reshape(1, n, n), C))
