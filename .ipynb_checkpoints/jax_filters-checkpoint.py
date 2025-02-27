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
    _, _, m_prev, C_prev = m_C_prev
    m_pred = M @ m_prev
    C_pred = M @ C_prev @ M.T + Q
    m_update = (jnp.eye(n) - K @ H) @ m_pred + K @ y_curr
    C_update = (jnp.eye(n) - K @ H) @ C_pred @ (jnp.eye(n) - K @ H).T + K @ R @ K.T #+ K @ H @ Q @ (K @ H).T
    
    return (m_pred, C_pred, m_update, C_update), (m_pred, C_pred, m_update, C_update)


@partial(jit, static_argnums=(4))
def apply_filtering_fixed_linear(m0, C0, y, K, n, M, H, Q, R):
    """
    Applies the filtering process to estimate the system state over time.
    Returns:
    m: Estimated states (mean) over time.
    C: Covariance matrices of the state estimates over time.
    """
    partial_filter_step = lambda m_C_prev, y_curr: filter_step_linear(m_C_prev, y_curr, K, n, M, H, Q, R)
    _, m_C = lax.scan(partial_filter_step, (m0, C0, m0, C0), y)
    m_preds, C_preds, m_updates, C_updates = m_C
    return m_preds, C_preds, m_updates, C_updates

@partial(jit, static_argnums=(3))
def filter_step_nonlinear(m_C_prev, y_curr, K, n, state_transition_function, jacobian_function, H, Q, R):
    """
    Apply a single forecast and Kalman filter step for a non-linear model.
    Returns:
    Tuple of updated state estimate (mean) and covariance for both the return and next step.
    """
    _, _, m_prev, C_prev = m_C_prev
    m_pred = state_transition_function(m_prev)
    F_jac = jacobian_function(m_prev)
    m_update = (jnp.eye(n) - K @ H) @ m_pred + K @ y_curr
    C_pred = F_jac @ C_prev @ F_jac.T + Q
    C_update = (jnp.eye(n) - K @ H) @ C_pred @ (jnp.eye(n) - K @ H).T + K @ R @ K.T #no discard yet
    return (m_pred, C_pred, m_update, C_update), (m_pred, C_pred, m_update, C_update)


@partial(jit, static_argnums=(4))
def apply_filtering_fixed_nonlinear(m0, C0, y, K, n, state_transition_function, jacobian_function, H, Q, R):
    """
    Applies the filtering process to estimate the system state over time for a non-linear model.
    """
    partial_filter_step = lambda m_C_prev, y_curr: filter_step_nonlinear(m_C_prev, y_curr, K, n, state_transition_function, jacobian_function, H, Q, R)
    _, m_C = lax.scan(partial_filter_step, (m0, C0, m0, C0), y)
    m_preds, C_preds, m_updates, C_updates = m_C
    return m_preds, C_preds, m_updates, C_updates


@partial(jit, static_argnums=(3))
def filter_step_nonlinear_H(m_C_prev, y_curr, K, n, state_transition_function, jacobian_function, observation_function, observation_jacobian, Q, R):
    """
    Apply a single forecast and Kalman filter step for a non-linear model.
    """
    _, _, m_prev, C_prev = m_C_prev
    m_pred = state_transition_function(m_prev)
    F_jac = jacobian_function(m_prev)
    
    # Nonlinear observation
    H_x_pred = observation_function(m_pred)
    H_jac = observation_jacobian(m_pred)  # Jacobian of H(x)

    # Compute predicted covariance
    C_pred = F_jac @ C_prev @ F_jac.T + Q
    S = H_jac @ C_pred @ H_jac.T + R  # Innovation covariance
    K_curr = C_pred @ H_jac.T @ jnp.linalg.inv(S)  # Kalman Gain

    # Update state estimate and covariance
    m_update = m_pred + K_curr @ (y_curr - H_x_pred)
    C_update = (jnp.eye(n) - K_curr @ H_jac) @ C_pred

    return (m_pred, C_pred, m_update, C_update), (m_pred, C_pred, m_update, C_update)

@partial(jit, static_argnums=(4))
def apply_filtering_fixed_nonlinear_H(m0, C0, y, K, n, state_transition_function, jacobian_function, observation_function, observation_jacobian, Q, R):
    """
    Applies the filtering process to estimate the system state over time for a non-linear model.
    """
    partial_filter_step = lambda m_C_prev, y_curr: filter_step_nonlinear(
        m_C_prev, y_curr, K, n, state_transition_function, jacobian_function, observation_function, observation_jacobian, Q, R
    )
    _, m_C = lax.scan(partial_filter_step, (m0, C0, m0, C0), y)
    m_preds, C_preds, m_updates, C_updates = m_C
    return m_preds, C_preds, m_updates, C_updates


@jit
def ledoit_wolf(P, shrinkage):
    return (1 - shrinkage) * P + shrinkage * jnp.trace(P)/P.shape[0] * jnp.eye(P.shape[0])

@jit
def sqrtm(M):
    eigenvalues, eigenvectors = jnp.linalg.eigh(M)
    inv_sqrt_eigenvalues = jnp.sqrt(eigenvalues)
    Lambda_inv_sqrt = jnp.diag(inv_sqrt_eigenvalues)
    M_sqrt = eigenvectors @ Lambda_inv_sqrt @ eigenvectors.T
    return M_sqrt.real


@jit
def ensrf_step(ensemble, y, H, Q, R, localization_matrix, inflation, key):
    n_ensemble = ensemble.shape[1]
    x_m = jnp.mean(ensemble, axis=1)
    raw_A = (ensemble - x_m.reshape((-1, 1))) 
    C_pred = (raw_A @ raw_A.T) / (n_ensemble - 1) + Q
    C_pred = ledoit_wolf(C_pred, 0.1)
    A = raw_A * inflation
    P = localization_matrix*(A @ A.T) / (n_ensemble - 1) + Q
    K = P @ H.T @ jnp.linalg.inv(H @ P @ H.T + R)
    x_m += K @ (y - H @ x_m)
    M_sqrt = sqrtm(jnp.eye(x_m.shape[0]) - K@H)
    updated_A = M_sqrt @ A
    updated_ensemble = x_m.reshape((-1, 1)) + updated_A
    updated_P = (updated_A @ updated_A.T / (n_ensemble - 1))
    updated_P = ledoit_wolf(updated_P, 0.1) #shrinkage
    return ensemble, C_pred, updated_ensemble, updated_P


@partial(jit, static_argnums=(2))
def ensrf_steps(state_transition_function, ensemble_init, num_steps, observations, observation_interval, H, Q, R, localization_matrix, inflation, key):
    model_vmap = jax.vmap(lambda v: state_transition_function(v), in_axes=1, out_axes=1)
    key, *subkeys = random.split(key, num=num_steps + 1)
    subkeys = jnp.array(subkeys)

    def inner(carry, t):
        ensemble, covar = carry
        ensemble_predicted = model_vmap(ensemble)
        def true_fun(_):
            x_m, C_pred, ensemble_updated, Pf_updated = ensrf_step(ensemble_predicted, observations[t, :], H, Q, R, localization_matrix, inflation, subkeys[t])
            return x_m, C_pred, ensemble_updated, Pf_updated
        def false_fun(_): # will require an update for larger observation intervals
            return ensemble_predicted, covar, ensemble_predicted, covar
        _, C_pred, ensemble_updated, Pf_updated = lax.cond(t % observation_interval == 0, true_fun, false_fun, operand=None)
        return (ensemble_updated, Pf_updated), (ensemble_predicted, C_pred, ensemble_updated, Pf_updated)

    n = len(Q[0])
    covariance_init = jnp.zeros((n, n))
    _, (ensemble_preds, C_preds, ensembles, covariances) = jax.lax.scan(inner, (ensemble_init, covariance_init), jnp.arange(num_steps))

    return ensemble_preds, C_preds, ensembles, covariances

@jit
def ensrf_step_H(ensemble, y, observation_function, observation_jacobian, Q, R, localization_matrix, inflation, key):
    n_ensemble = ensemble.shape[1]
    x_m = jnp.mean(ensemble, axis=1)
    raw_A = (ensemble - x_m.reshape((-1, 1))) 

    # Compute predicted covariance
    C_pred = (raw_A @ raw_A.T) / (n_ensemble - 1) + Q
    C_pred = ledoit_wolf(C_pred, 0.1)
    
    A = raw_A * inflation
    P = localization_matrix * (A @ A.T) / (n_ensemble - 1) + Q
    
    # Apply nonlinear observation
    H_x_pred = observation_function(x_m)
    H_jac = observation_jacobian(x_m)  # Compute Jacobian for Kalman gain
    
    S = H_jac @ P @ H_jac.T + R  # Innovation covariance
    K = P @ H_jac.T @ jnp.linalg.inv(S)  # Kalman gain

    x_m += K @ (y - H_x_pred)
    
    M_sqrt = sqrtm(jnp.eye(x_m.shape[0]) - K @ H_jac)
    updated_A = M_sqrt @ A
    updated_ensemble = x_m.reshape((-1, 1)) + updated_A
    
    updated_P = (updated_A @ updated_A.T / (n_ensemble - 1))
    updated_P = ledoit_wolf(updated_P, 0.1)  # Shrinkage

    return ensemble, C_pred, updated_ensemble, updated_P

@partial(jit, static_argnums=(2))
def ensrf_steps_H(state_transition_function, ensemble_init, num_steps, observations, observation_interval, observation_function, observation_jacobian, Q, R, localization_matrix, inflation, key):
    model_vmap = jax.vmap(lambda v: state_transition_function(v), in_axes=1, out_axes=1)
    key, *subkeys = random.split(key, num=num_steps + 1)
    subkeys = jnp.array(subkeys)

    def inner(carry, t):
        ensemble, covar = carry
        ensemble_predicted = model_vmap(ensemble)
        
        def true_fun(_):
            x_m, C_pred, ensemble_updated, Pf_updated = ensrf_step_H(
                ensemble_predicted, observations[t, :], observation_function, observation_jacobian, Q, R, localization_matrix, inflation, subkeys[t]
            )
            return x_m, C_pred, ensemble_updated, Pf_updated
        
        def false_fun(_):
            return ensemble_predicted, covar, ensemble_predicted, covar

        _, C_pred, ensemble_updated, Pf_updated = lax.cond(t % observation_interval == 0, true_fun, false_fun, operand=None)
        
        return (ensemble_updated, Pf_updated), (ensemble_predicted, C_pred, ensemble_updated, Pf_updated)

    n = len(Q[0])
    covariance_init = jnp.zeros((n, n))
    _, (ensemble_preds, C_preds, ensembles, covariances) = jax.lax.scan(inner, (ensemble_init, covariance_init), jnp.arange(num_steps))

    return ensemble_preds, C_preds, ensembles, covariances

# this the generalized nonlinear filter, with observation interval functionality
@jit
def kalman_filter_process(state_transition_function, jacobian_function, m0, C0, observations, H, Q, R, observation_interval):
    params = (jacobian_function, H, Q, R)
    initial_state = (m0, C0)

    def step_fn(carry, t):
        m_prev, C_prev = carry
        m_pred = state_transition_function(m_prev)
        F_jac = jacobian_function(m_prev)
        C_pred = F_jac @ C_prev @ F_jac.T + Q

        def true_fun(_):
            S = H @ C_pred @ H.T + R
            K_curr = C_pred @ H.T @ jnp.linalg.inv(S)
            m_update = m_pred + K_curr @ (observations[t, :] - H @ m_pred)
            C_update = (jnp.eye(H.shape[1]) - K_curr @ H) @ C_pred
            return (m_update, C_update), (m_pred, C_pred, m_update, C_update, K_curr)

        def false_fun(_):
            return (m_pred, C_pred), (m_pred, C_pred, m_pred, C_pred, jnp.zeros((C_pred.shape[0], H.shape[0])))

        return lax.cond(t % observation_interval == 0, true_fun, false_fun, operand=None)

    _, (m_preds, C_preds, m_updates, C_updates, Ks) = lax.scan(
        step_fn, initial_state, jnp.arange(observations.shape[0])
    )

    return m_preds, C_preds, m_updates, C_updates, Ks

@jit
def kalman_step_H(state, observation, params):
    """
    Performs a single Kalman filter update step using a nonlinear observation function H(x).
    """
    m_prev, C_prev = state
    state_transition_function, jacobian_function, observation_function, observation_jacobian, Q, R = params
    
    # Prediction step
    m_pred = state_transition_function(m_prev)
    F_jac = jacobian_function(m_prev)
    C_pred = F_jac @ C_prev @ F_jac.T + Q
    
    # Compute nonlinear observation and its Jacobian
    H_x_pred = observation_function(m_pred)  # Apply H(x)
    H_jac = observation_jacobian(m_pred)  # Compute Jacobian of H(x)

    # Update step
    S = H_jac @ C_pred @ H_jac.T + R  # Innovation covariance
    K_curr = C_pred @ H_jac.T @ jnp.linalg.inv(S)  # Kalman gain
    m_update = m_pred + K_curr @ (observation - H_x_pred)  # Update mean
    C_update = (jnp.eye(H_jac.shape[1]) - K_curr @ H_jac) @ C_pred  # Update covariance
    
    return (m_update, C_update), (m_update, C_update, K_curr)

@jit
def kalman_filter_process_H(state_transition_function, jacobian_function, m0, C0, observations, observation_function, observation_jacobian, Q, R):
    """
    Extended Kalman Filter process with nonlinear observation function H(x).
    """
    params = (state_transition_function, jacobian_function, observation_function, observation_jacobian, Q, R)
    initial_state = (m0, C0)
    
    _, (m_preds, C_preds, m_updates, C_updates, Ks) = lax.scan(
        lambda state, obs: kalman_step_H(state, obs, params),
        initial_state, 
        observations
    )
    
    return m_preds, C_preds, m_updates, C_updates, Ks


@jit
def resample_particles(key, particles, weights):
    num_particles = particles.shape[0]
    cumulative_sum = jnp.cumsum(weights)
    indices = jnp.searchsorted(cumulative_sum, random.uniform(key, (num_particles,)))
    return particles[indices]

@jit
def update_weights(particles, observation, H, R):
    # Calculate the likelihood of each particle given the observation
    predicted_observations = jax.vmap(lambda x: jnp.dot(H, x), in_axes=0, out_axes=0)(particles)
    obs_dim = observation.shape[0]
    inv_R = jnp.linalg.inv(R)
    diff = observation - predicted_observations
    likelihood = jnp.exp(-0.5 * jax.vmap(lambda d: jnp.dot(d, jnp.dot(inv_R, d.T)), in_axes=0, out_axes=0)(diff))
    likelihood = likelihood / likelihood.sum()  # Normalize the weights
    return likelihood

@partial(jax.jit, static_argnums=(1, 2, 5))  
def particle_filter(key, num_particles, num_steps, initial_state, observations, observation_interval, 
                    state_transition_function, H, Q, R):
    mean = jnp.tile(initial_state, (num_particles, 1))
    particles = random.multivariate_normal(key, mean, Q, shape=(num_particles,))
    step = jax.vmap(state_transition_function, in_axes=0, out_axes=0)

    def body_fn(carry, t):
        key, particles = carry
        key, subkey = random.split(key)
        particles = step(particles) + random.multivariate_normal(subkey, jnp.zeros(particles.shape[1]), Q, shape=(num_particles,))
        # Only update weights and resample at observation intervals
        def update_and_resample(particles):
            obs_idx = t // observation_interval
            observation = observations[obs_idx]
            weights = update_weights(particles, observation, H, R)
            return resample_particles(subkey, particles, weights)
        particles = lax.cond(t % observation_interval == 0, update_and_resample, lambda x: x, particles)
        return (key, particles), particles
    _, ensemble = lax.scan(body_fn, (key, particles), jnp.arange(num_steps))
    
    return jnp.transpose(ensemble, (0, 2, 1))  # (timestep, state_dim, num_particles)

