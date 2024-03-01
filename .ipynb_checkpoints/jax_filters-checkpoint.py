import jax
import jax.numpy as jnp
from jax import random, jit, lax
from jax.scipy.linalg import inv, svd, eigh, det
from jax.numpy.fft import fft, ifft
from functools import partial

@jit
def filter_step_linear(m_C_prev, y_curr, K, n, M, H, Q, R):
    """
    Apply a single forecast and Kalman filter step with fixed gain.
    Tuple of updated state estimate (mean) and covariance, both for return and for next step.
    """
    m_prev, C_prev = m_C_prev
    m_pred = M @ m_prev
    C_pred = M @ C_prev @ M.T + Q
    m_update = (jnp.eye(n) - K @ H) @ m_pred + K @ y_curr
    C_update = (jnp.eye(n) - K @ H) @ C_pred @ (jnp.eye(n) - K @ H).T + K @ R @ K.T
    
    return (m_update, C_update), (m_update, C_update)

@jit
def apply_filtering_fixed_linear(m0, C0, y, K, n, M, H, Q, R):
    """
    Applies the filtering process to estimate the system state over time.
    Returns:
    m: Estimated states (mean) over time.
    C: Covariance matrices of the state estimates over time.
    """
    partial_filter_step = lambda m_C_prev, y_curr: filter_step_linear(m_C_prev, y_curr, K, n, M, H, Q, R)
    _, m_C = lax.scan(partial_filter_step, (m0, C0), y)
    m, C = m_C
    return jnp.vstack((m0[None, :], m)), jnp.concatenate((C0[None, :, :], C), axis=0)

@jit
def filter_step(m_C_prev, y_curr, K, n, state_transition_function, jacobian_function, H, Q, R):
    """
    Apply a single forecast and Kalman filter step for a non-linear model.
    Returns:
    Tuple of updated state estimate (mean) and covariance for both the return and next step.
    """
    m_prev, C_prev = m_C_prev
    m_pred = state_transition_function(m_prev)
    F_jac = jacobian_function(m_prev)
    m_update = (jnp.eye(n) - K @ H) @ m_pred + K @ y_curr
    C_pred = F_jac @ C_prev @ F_jac.T + Q
    C_update = (jnp.eye(n) - K @ H) @ C_pred @ (jnp.eye(n) - K @ H).T + K @ R @ K.T
    return (m_update, C_update), (m_update, C_update)

@jit
def apply_filtering_fixed_nonlinear(m0, C0, y, K, n, state_transition_function, jacobian_function, H, Q, R):
    """
    Applies the filtering process to estimate the system state over time for a non-linear model.
    """
    partial_filter_step = lambda m_C_prev, y_curr: filter_step(m_C_prev, y_curr, K, n, state_transition_function, jacobian_function, H, Q, R)
    _, m_C = lax.scan(partial_filter_step, (m0, C0), y)
    m, C = m_C
    return jnp.vstack((m0[None, :], m)), jnp.concatenate((C0[None, :, :], C), axis=0)

@jit
def old_ensrf_step(ensemble, y, H, Q, R, localization_matrix, inflation):
    n_ensemble = ensemble.shape[1]
    x_m = jnp.mean(ensemble, axis=1)
    I = jnp.eye(x_m.shape[0])
    A = ensemble - x_m.reshape((-1, 1))
    Pf = inflation * A @ A.T / (n_ensemble - 1)
    P = Pf * localization_matrix + Q  # Element-wise multiplication for localization
    K = P @ H.T @ jnp.linalg.inv(H @ P @ H.T + R)
    P_updated = (I - K @ H) @ P
    x_m += K @ (y - H @ x_m)
    M = I + P @ H.T @ jnp.linalg.inv(R) @ H
    eigenvalues, eigenvectors = eigh(M)
    inv_sqrt_eigenvalues = 1 / jnp.sqrt(eigenvalues)
    Lambda_inv_sqrt = jnp.diag(inv_sqrt_eigenvalues)
    M_inv_sqrt = eigenvectors @ Lambda_inv_sqrt @ eigenvectors.T
    updated_ensemble = x_m.reshape((-1, 1)) + M_inv_sqrt @ A
    return updated_ensemble, P_updated

@jit
def ensrf_step(ensemble, y, H, Q, R, localization_matrix, inflation, key):
    n_ensemble = ensemble.shape[1]
    x_m = jnp.mean(ensemble, axis=1)
    ensemble += random.multivariate_normal(key, jnp.zeros(ensemble.shape[0]), Q, (n_ensemble,)).T
    A = ensemble - x_m.reshape((-1, 1))
    Pf = inflation*(A @ A.T) / (n_ensemble - 1)
    P = Pf * localization_matrix  # Element-wise multiplication for localization
    K = P @ H.T @ jnp.linalg.inv(H @ P @ H.T + R)
    x_m += K @ (y - H @ x_m)
    M = jnp.eye(x_m.shape[0]) + P @ H.T @ jnp.linalg.inv(R) @ H
    eigenvalues, eigenvectors = eigh(M)
    inv_sqrt_eigenvalues = 1 / jnp.sqrt(eigenvalues)
    Lambda_inv_sqrt = jnp.diag(inv_sqrt_eigenvalues)
    M_inv_sqrt = eigenvectors @ Lambda_inv_sqrt @ eigenvectors.T
    updated_ensemble = x_m.reshape((-1, 1)) + M_inv_sqrt @ A
    updated_A = updated_ensemble - jnp.mean(updated_ensemble, axis=1).reshape((-1, 1))
    updated_P = localization_matrix*(updated_A @ updated_A.T / (n_ensemble - 1))
    return updated_ensemble, updated_P + jnp.eye(x_m.shape[0])*1e-5  #adds matrix to keep psd


@partial(jit, static_argnums=(3))

def ensrf_steps(state_transition_function, n_ensemble, ensemble_init, num_steps, observations, observation_interval, H, Q, R, localization_matrix, inflation, key):
    """
    Deterministic Ensemble Square Root Filter generalized for any model.
    """
    model_vmap = jax.vmap(lambda v: state_transition_function(v), in_axes=1, out_axes=1)
    key, *subkeys = random.split(key, num=num_steps+1)
    subkeys = jnp.array(subkeys)
    def inner(carry, t):
        ensemble, previous_covariance = carry
        ensemble_predicted = model_vmap(ensemble)    
        def true_fun(_):
            return ensrf_step(ensemble_predicted, observations[t, :], H, Q, R, localization_matrix, inflation, subkeys[t])  
        def false_fun(_):# Use the last updated covariance if no observation is available
            return ensemble_predicted, previous_covariance
        ensemble_updated, Pf_updated = lax.cond(t % observation_interval == 0, true_fun, false_fun, operand=None)
        return (ensemble_updated, Pf_updated), (ensemble_updated, Pf_updated)
    n = len(H[0])
    covariance_init = jnp.zeros((n,n))
    _, output = jax.lax.scan(inner, (ensemble_init, covariance_init), jnp.arange(num_steps))
    ensembles, covariances = output
    return ensembles, covariances

@jit
def kalman_filter_process(state_transition_function, jacobian_function, m0, C0, observations, H, Q, R, dt):
    """
    Runs the Kalman filter process to update the state estimate and Kalman gain K at each step.
    """
    n = m0.shape[0]
    num_steps = observations.shape[0]
    m = jnp.zeros((num_steps, n))
    C = jnp.zeros((num_steps, n, n))
    K = jnp.zeros((num_steps, n, n))
    m_prev = m0
    C_prev = C0

    for i in range(num_steps):
        m_pred = state_transition_function(m_prev)
        F_jac = jacobian_function(m_pred)
        C_pred = F_jac @ C_prev @ F_jac.T + Q
        S = H @ C_pred @ H.T + R
        K_curr = C_pred @ H.T @ jnp.linalg.inv(S)
        m_update = m_pred + K_curr @ (observations[i] - H @ m_pred)
        C_update = (jnp.eye(n) - K_curr @ H) @ C_pred
        m = m.at[i, :].set(m_update)
        C = C.at[i, :, :].set(C_update)
        K = K.at[i, :, :].set(K_curr)
        m_prev = m_update
        C_prev = C_update

    return m, C, K
