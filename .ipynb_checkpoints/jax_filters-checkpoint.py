import jax.numpy as np
from jax import random
from jax import jit
from jax.scipy.linalg import svd
from jax.numpy.fft import fft, ifft
import jax

from jax import jit, numpy as jnp
from jax.lax import scan
from jax import jit, numpy as jnp
from jax.lax import scan

@jit
def filter_step_linear(m_C_prev, y_curr, K,n, M, H, Q, R):
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
    # Define a partial function to include all parameters except the one iterated over by scan
    partial_filter_step = lambda m_C_prev, y_curr: filter_step_linear(m_C_prev, y_curr, K, n, M, H, Q, R)
    
    # Apply scan to iterate over all observations
    _, m_C = scan(partial_filter_step, (m0, C0), y)
    
    # Extract mean and covariance from the result
    m, C = m_C
    
    # Stack initial conditions to the results for complete trajectories
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
    
    Args:
    y: Observations over time.
    K: Kalman gain matrix.
    state_transition_function: Function to predict next state from the current state.
    jacobian_function: Function to compute the Jacobian of the state transition function.
    dt: Time step size.
    F: Parameter for the state transition function.
    H: Observation matrix.
    Q: Process noise covariance.
    R: Observation noise covariance.
    m0: Initial state estimate (mean).
    C0: Initial covariance estimate.
    n: Dimension of the state space.
    
    Returns:
    m: Estimated states (mean) over time.
    C: Covariance matrices of the state estimates over time.
    """
    # Define a partial function to include all necessary parameters
    partial_filter_step = lambda m_C_prev, y_curr: filter_step(m_C_prev, y_curr, K, n, state_transition_function, jacobian_function, H, Q, R)
    # Apply scan to iterate over all observations
    _, m_C = scan(partial_filter_step, (m0, C0), y)
    m, C = m_C
    # Stack initial conditions with the results for complete trajectories
    return jnp.vstack((m0[None, :], m)), jnp.concatenate((C0[None, :, :], C), axis=0)


@jit
def ensrf_step(ensemble, n_ensemble, y, H, Q, R, localization_matrix, inflation):
    x_m = np.mean(ensemble, axis=1)
    A = ensemble - x_m.reshape((-1, 1))
    Pf = inflation*A@A.T/(n_ensemble - 1)
    #print(Pf) rank deficient when ensemble members less than state dimension (can't invert, determinant 0)
    P = Pf * localization_matrix + Q # Element-wise multiplication for localization
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    # Update ensemble states
    x_m += K@(y - H@x_m)
    M = np.eye(x_m.shape[0]) + P @ H.T @ np.linalg.inv(R) @ H
    U, s, Vh = svd(M)
    s_inv_sqrt = np.diag(s**-0.5)
    M_inv_sqrt = U @ s_inv_sqrt @ Vh
    ensemble = x_m.reshape((-1, 1)) + np.real(M_inv_sqrt) @ A

    #ensemble = x_m.reshape((-1, 1)) + np.real(np.linalg.inv(sqrtm(np.eye(x_m.shape[0]) + P@H.T@np.linalg.inv(R)@H)))@A
    return ensemble #can pull covariance out of this, but should apply localization

def ensrf_steps(model, n_ensemble, ensemble_init, n_timesteps, observations, observation_interval, H, Q, R, localization_matrix, inflation):
    """
    Deterministic Ensemble Square Root Filter generalized for any model.

    :param model: Model function.
    :param state_dim: Dimension of state.
    :param n_ensemble: Number of ensemble members.
    :param n_timesteps: Number of time steps.
    :param dt: Time step size.
    :param H: Observation matrix.
    :param observations: Observations at each time step.
    :param Q: Model noise covariance matrix.
    :return: Array containing states.
    """
    model_vmap = jax.vmap(lambda v: model(v), in_axes=1, out_axes=1)

    def inner(ensemble, t):
        ensemble = model_vmap(ensemble)

        ensemble = jax.lax.cond(t % observation_interval == 0,
                                lambda _: ensrf_step(ensemble, n_ensemble,observations[t, :], H, Q, R,localization_matrix, inflation),
                                lambda _: ensemble, 
                                None)

        return ensemble, ensemble

    _, states = jax.lax.scan(inner, ensemble_init, np.arange(n_timesteps))

    return states


def kalman_filter_process(state_transition_function, jacobian_function, m0, C0, observations, H, Q, R, dt):
    """
    Runs the Kalman filter process to update the state estimate and Kalman gain K at each step.
    Args:
    H: Observation matrix.
    Q: Process noise covariance matrix.
    R: Measurement noise covariance matrix.
    m0: Initial state estimate.
    C0: Initial covariance estimate.
    observations: Observations over time.
    dt: Time step for state transition.

    Returns:
    m: Updated state estimates over time.
    C: Updated covariance matrices over time.
    K: Updated Kalman gain matrices over time.
    """
    n = m0.shape[0]
    num_steps = observations.shape[0]

    m = jnp.zeros((num_steps, n))
    C = jnp.zeros((num_steps, n, n))
    K = jnp.zeros((num_steps, n, n))

    m_prev = m0
    C_prev = C0

    for i in range(num_steps):
        # Prediction Step
        m_pred = state_transition_function(m_prev)
        F_jac = jacobian_function(m_pred)
        C_pred = F_jac @ C_prev @ F_jac.T + Q

        # Update Step
        S = H @ C_pred @ H.T + R
        K_curr = C_pred @ H.T @ jnp.linalg.inv(S)
        m_update = m_pred + K_curr @ (observations[i] - H @ m_pred)
        C_update = (jnp.eye(n) - K_curr @ H) @ C_pred

        # Store results
        m = m.at[i, :].set(m_update)
        C = C.at[i, :, :].set(C_update)
        K = K.at[i, :, :].set(K_curr)

        m_prev = m_update
        C_prev = C_update

    return m, C, K