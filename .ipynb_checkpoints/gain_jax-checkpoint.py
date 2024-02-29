# from jax import config
# config.update("jax_debug_nans", True)
import jax.numpy as jnp
from jax import random, grad, jit
from jax.scipy.linalg import inv, det
from jax.lax import scan
import jax
from scipy.linalg import solve_discrete_are
from tqdm import tqdm
import matplotlib.pyplot as plt

key = random.PRNGKey(3)

# System dimensions
n = 2  # System dimension
p = 2  # Observation dimension
J = 1000 # number of steps
J0 = 0 # burn in period
N = 10 # Monte Carlo samples

# Model parameters
m0 = jnp.zeros((n,))  # Initial state mean
C0 = jnp.eye(n) * 1.0   # Initial state covariance matrix (P)
Q = jnp.eye(n) * 5.0    # Process noise covariance matrix (Sigma in Julia code)
R = jnp.eye(n) * 1.0    # Observation noise covariance matrix (Gamma)
inv_R = inv(R)
M = jnp.eye(n) * 0.9    # State transition matrix (A)
H = jnp.eye(n)          # Observation matrix

# State initialization
vd0 = m0 + random.multivariate_normal(key, jnp.zeros(n), C0)
vd = jnp.zeros((J, n))
vd = vd.at[1].set(vd0)

key, _ = random.split(key)

# State update
for j in range(1, J):
    key, _ = random.split(key)
    vd = vd.at[j].set(M @ vd[j-1] + random.multivariate_normal(key, jnp.zeros(n), Q))

key, _ = random.split(key)

# Observation generation
noise = random.multivariate_normal(key, jnp.zeros(p), R, (J,))
y = vd + noise  # Transpose the noise to match the shape of vd

@jit
def filter_step(m_C_prev, y_curr, K):
    """
    Apply a single forecast and Kalman filter step.
    """
    m_prev, C_prev = m_C_prev
    m_pred = M @ m_prev
    m_update = (jnp.eye(n) - K @ H) @ m_pred + K @ y_curr
    C_pred = M @ C_prev @ M.T + Q
    C_update = (jnp.eye(n) - K @ H) @ C_pred @ (jnp.eye(n) - K @ H).T + K @ R @ K.T
    return (m_update, C_update), (m_update, C_update)

@jit
def filtered(K):
    """
    Applies the filtering process to estimate the system state.

    Args:
    K: Kalman gain matrix.

    Returns:
    m: Estimated states over time.
    C: Covariance matrices of the state estimates over time.
    """
    _, m_C = scan(lambda m_C_prev, y_curr: filter_step(m_C_prev, y_curr, K), (m0, C0), y)
    m, C = m_C
    return jnp.vstack((m0, m)), jnp.vstack((C0.reshape(1, n, n), C))

@jit
def KL_gaussian(m1, C1, m2, C2):
    """
    Computes the Kullback-Leibler divergence between two Gaussian distributions.
    m1, C1: Mean and covariance of the first Gaussian distribution.
    m2, C2: Mean and covariance of the second Gaussian distribution.
    """
    C2_inv = inv(C2)
    return 0.5 * (jnp.log(det(C2) / det(C1)) - n + jnp.trace(C2_inv @ C1) + ((m2 - m1).T @ C2_inv @ (m2 - m1)))

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
    return -0.5 * sum_ll - 0.5 * (J - J0) * p * jnp.log(2 * jnp.pi) - 0.5 * (J - J0) * jnp.log(det(R))

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
        key, *subkeys_inner = random.split(key, num=N+1)

        def inner_map(subkey):
            v_pred = M@(m_prev + random.multivariate_normal(subkey, jnp.zeros(n), C_prev))
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
    key, *subkeys = random.split(key, num=N+1)
    def inner_map(subkey):
        return log_likelihood(random.multivariate_normal(subkey, m, C), y)
    return (KL_sum(m, C, K, key) - jnp.mean(jax.lax.map(inner_map, jnp.vstack(subkeys))))

# Steady state gain and optimization

P = solve_discrete_are(M.T, H.T, Q, R)
# Compute steady-state Kalman gain K
K_steady = P @ H.T @ jnp.linalg.inv(H @ P @ H.T + R)
print("Steady-state K:", K_steady)

# Define the gradient of the cost function
var_cost_grad = grad(var_cost, argnums=0)

# Initial guess for K and optimization parameters
K_opt = jnp.eye(n) * 0.4
alpha = 1e-5
errs = []
for i in tqdm(range(100)):
    key, _ = random.split(key)
    K_opt -= alpha * var_cost_grad(K_opt, key)
    errs.append(jnp.linalg.norm(K_opt - K_steady))
print("Optimized K:", K_opt)
print("Steady-state K:", K_steady)
print("Error:", jnp.linalg.norm(K_opt - K_steady))

plt.plot(errs)