import jax
import jax.numpy as jnp
from jax import random, jit, lax
from jax.scipy.linalg import inv, svd, eigh, det
from jax.numpy.fft import fft, ifft
from functools import partial

@partial(jit, static_argnums=(3))
def filter_step_linear(m_C_prev, y_curr, K, n, M, H, Q, R):
    """
    Apply a single forecast and Kalman filter step with fixed gain.
    Tuple of updated state estimate (mean) and covariance, both for return and for next step.
    """
    m_prev, C_prev = m_C_prev
    m_pred = M @ m_prev
    C_pred = M @ C_prev @ M.T + Q
    m_update = (jnp.eye(n) - K @ H) @ m_pred + K @ y_curr
    C_update = (jnp.eye(n) - K @ H) @ C_pred @ (jnp.eye(n) - K @ H).T + K @ R @ K.T #we discard this term if we want to look for covariance wrt true filter, not wrt truth
    
    return (m_update, C_update), (m_update, C_update)


@partial(jit, static_argnums=(4))
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
    return m, C

@partial(jit, static_argnums=(3))
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
    C_update = (jnp.eye(n) - K @ H) @ C_pred @ (jnp.eye(n) - K @ H).T + K @ R @ K.T #no discard yet
    return (m_update, C_update), (m_update, C_update)


@partial(jit, static_argnums=(4))
def apply_filtering_fixed_nonlinear(m0, C0, y, K, n, state_transition_function, jacobian_function, H, Q, R):
    """
    Applies the filtering process to estimate the system state over time for a non-linear model.
    """
    partial_filter_step = lambda m_C_prev, y_curr: filter_step(m_C_prev, y_curr, K, n, state_transition_function, jacobian_function, H, Q, R)
    _, m_C = lax.scan(partial_filter_step, (m0, C0), y)
    m, C = m_C
    return m, C


@jit
def ledoit_wolf(P, shrinkage):
    return (1 - shrinkage) * P + shrinkage * jnp.trace(P)/P.shape[0] * jnp.eye(P.shape[0])

@jit
def invsqrtm(M):
    eigenvalues, eigenvectors = jnp.linalg.eigh(M)
    inv_sqrt_eigenvalues = jnp.sqrt(eigenvalues)
    Lambda_inv_sqrt = jnp.diag(inv_sqrt_eigenvalues)
    M_inv_sqrt = eigenvectors @ Lambda_inv_sqrt @ eigenvectors.T
    return M_inv_sqrt.real


@jit
def ensrf_step(ensemble, y, H, Q, R, localization_matrix, inflation, key):
    n_ensemble = ensemble.shape[1]
    x_m = jnp.mean(ensemble, axis=1)
    A = ensemble - x_m.reshape((-1, 1))
    A = A*inflation
    Pf = (A @ A.T) / (n_ensemble - 1) + Q
    P = Pf * localization_matrix  # Element-wise multiplication for localization
    K = P @ H.T @ jnp.linalg.inv(H @ P @ H.T + R)
    x_m += K @ (y - H @ x_m)
    M_inv_sqrt = invsqrtm(jnp.eye(x_m.shape[0]) - K@H)
    updated_A = M_inv_sqrt @ A
    updated_ensemble = x_m.reshape((-1, 1)) + updated_A
    updated_P = (updated_A @ updated_A.T / (n_ensemble - 1))
    updated_P = ledoit_wolf(updated_P, 0.1) #shrinkage
    return updated_ensemble, updated_P


@partial(jit, static_argnums=(3))
def ensrf_steps(state_transition_function, n_ensemble, ensemble_init, num_steps, observations, observation_interval, H, Q, R, localization_matrix, inflation, key):
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
def kalman_step(state, observation, params):
    m_prev, C_prev = state
    state_transition_function, jacobian_function, H, Q, R = params
    m_pred = state_transition_function(m_prev)
    F_jac = jacobian_function(m_prev)
    C_pred = F_jac @ C_prev @ F_jac.T + Q
    S = H @ C_pred @ H.T + R
    K_curr = C_pred @ H.T @ jnp.linalg.inv(S)
    m_update = m_pred + K_curr @ (observation - H @ m_pred)
    C_update = (jnp.eye(H.shape[1]) - K_curr @ H) @ C_pred
    
    return (m_update, C_update), (m_update, C_update, K_curr)

@jit
def kalman_filter_process(state_transition_function, jacobian_function, m0, C0, observations, H, Q, R):
    params = (state_transition_function, jacobian_function, H, Q, R)
    initial_state = (m0, C0)

    # Execute `lax.scan` over the sequence of observations
    _, (m, C, K) = lax.scan(lambda state, obs: kalman_step(state, obs, params),
                            initial_state, observations)
    
    return m, C, K