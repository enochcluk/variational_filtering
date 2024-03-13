import jax
import jax.numpy as jnp
from jax import random, grad, jit, jacfwd, lax, vmap, jacrev
from jax.scipy.linalg import inv, eigh
from jax.lax import scan
from tqdm import tqdm
from functools import partial


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
    """
    C2_inv = inv(C2)
    log_det_ratio = (jnp.log(eigh(C2)[0]).sum() - jnp.log(eigh(C1)[0]).sum()).real 
    # log(det(C2) / det(C1)), works better with limited precision because the determinant is practically 0
    return 0.5 * (log_det_ratio - n + jnp.trace(C2_inv @ C1) + ((m2 - m1).T @ C2_inv @ (m2 - m1)))


@jit
def log_likelihood(v, y, H, R, J, J0):
    """
    Computes the log-likelihood of observations given state estimates as a sum.
    Compares this to R, observation noise
    """
    def log_likelihood_j(_, v_y):
        v_j, y_j = v_y
        error = y_j - H @ v_j
        ll = error.T @ inv(R) @ error
        return _, ll
    _, lls = lax.scan(log_likelihood_j, None, (v, y))
    sum_ll = jnp.nansum(lls)
    return -0.5 * sum_ll - 0.5 * (J - J0) * jnp.log(2 * jnp.pi) - 0.5 * (J - sum(jnp.isnan(lls)) - J0) * jnp.linalg.slogdet(R)[1]
    

@partial(jit, static_argnums=(2,6))
def KL_sum(m, C, n, state_transition_function, Q, key, N):
    """
    Computes the sum of KL divergences between the predicted and updated state distributions.
    """
    def KL_j(_, m_C_y):
        m_prev, m_curr, C_prev, C_curr, key = m_C_y
        key, *subkeys_inner = random.split(key, num=N)
        def inner_map(subkey):
            perturbed_state = m_prev + random.multivariate_normal(subkey, jnp.zeros(n), C_prev)
            v_pred = state_transition_function(perturbed_state)
            return KL_gaussian(n, m_curr, C_curr, v_pred, Q)
        mean_kl = jnp.mean(vmap(inner_map)(jnp.array(subkeys_inner)), axis=0)
        return _, mean_kl
    _, mean_kls = scan(KL_j, None, (m[:-1, :], m[1:, :], C[:-1, :, :], C[1:, :, :], jnp.array(random.split(key, num=m.shape[0]-1))))
    kl_sum = jnp.sum(mean_kls)
    return kl_sum

@jit
def var_cost(K, m0, C0, n, state_transition_function, Q, jacobian, H, R, y, key, N, J, J0, filtered_func, filter_step_func):
    """
    Computes the cost function for optimization, combining KL divergence and log-likelihood.
    """
    m, C = filtered_func(K, m0, C0, n, state_transition_function, Q, jacobian, H, R, y, filter_step_func)
    key, *subkeys = random.split(key, num=N+1)
    log_likelihood_vals = vmap(lambda subkey: log_likelihood(vmap(lambda x: random.multivariate_normal(subkey, x, C))(m), y, H, inv(R), R, J, J0))(jnp.array(subkeys))
    return (KL_sum(m, C, K, n, state_transition_function, Q, key, N) - jnp.mean(log_likelihood_vals))

@jit
def filter_step(m_C_prev, y_curr, K, n, state_transition_function, Q, jacobian, H, R):
    """
    Apply a single forecast and update step using the Kalman filter.
    """
    m_prev, C_prev = m_C_prev
    m_pred = state_transition_function(m_prev)
    F_jac = jacobian(m_prev)
    m_update = (jnp.eye(n) - K @ H) @ m_pred + K @ y_curr
    C_pred = F_jac @ C_prev @ F_jac.T + Q
    C_update = (jnp.eye(n) - K @ H) @ C_pred @ (jnp.eye(n) - K @ H).T + K @ R @ K.T
    return (m_update, C_update), (m_update, C_update)

@jit
def filtered(K, m0, C0, n, state_transition_function, Q, jacobian, H, R, y, filter_step_func):
    """
    Applies the filtering process to estimate the system state over time.
    """
    _, m_C = scan(lambda m_C_prev, y_curr: filter_step_func(m_C_prev, y_curr, K, n, state_transition_function, Q, jacobian, H, R), (m0, C0), y)
    m, C = m_C
    return jnp.vstack((m0[jnp.newaxis, :], m)), jnp.concatenate((C0[jnp.newaxis, :, :], C), axis=0)
