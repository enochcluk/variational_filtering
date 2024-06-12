import jax
import jax.numpy as jnp
from jax import random, grad, jit, jacfwd, lax, vmap, jacrev
from jax.scipy.linalg import inv, eigh
from jax.lax import scan
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt


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
    # replace log(det(C2) / det(C1)), works better with limited precision because the determinant is practically 0
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

@partial(jit, static_argnums=())
def KL_sum(m_preds, C_preds, m_updates, C_updates, n, state_transition_function, Q, key):
    """
    Computes the sum of KL divergences between the predicted and updated state distributions.
    """
    def KL_j(_, m_C_y):
        m_pred, C_pred, m_update, C_update = m_C_y
        return _, KL_gaussian(n, m_update, C_update, m_pred, C_pred)
    _, mean_kls = scan(KL_j, None, (m_preds, C_preds, m_updates, C_updates))
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


def plot_optimization_results(norms, prediction_errors, true_div, n_iters, file_path, scaling=1.3, max_n_locator=5):
    """
    norms (list): List of norm values representing norm from reference K.
    prediction_errors (list): List of prediction error values (MSE).
    true_div (list): List of KL divergence values (from Kalman Filter solution).
    n_iters (int): Number of iterations to plot.
    file_path (str): Path to save the plot as a PDF file.
    scaling (float): Scaling factor for font and label sizes.
    max_n_locator (int): Maximum number of labels on the y-axis.
    """
    fig, (ax1, ax3) = plt.subplots(figsize=(10, 4), ncols=2)
    
    # K norms and KL Divergence
    color = 'tab:red'
    ax1.set_xlabel('Iteration', fontsize=14*scaling)
    ax1.set_ylabel('Gain error ($\|K_\mathrm{opt} - K_\mathrm{steady}\|_F$)', color=color, fontsize=14*scaling)
    line1, = ax1.plot(range(1, n_iters+1), norms[:n_iters], label='Gain error', color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12*scaling)
    ax1.tick_params(axis='x', labelsize=12*scaling)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(max_n_locator))

    ax2 = ax1.twinx()
    color_pred = 'tab:green'
    ax2.set_ylabel('KL divergence to true filter', color=color_pred, fontsize=14*scaling)
    line2, = ax2.plot(range(1, n_iters+1), true_div[:n_iters], label='KL divergence to true filter', color=color_pred, linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color_pred, labelsize=12*scaling)
    ax2.tick_params(axis='x', labelsize=12*scaling)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(max_n_locator))

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, fontsize=8*scaling)

    # MSE from True States
    ax3.plot(range(1, n_iters+1), prediction_errors[:n_iters])
    ax3.set_xlabel("Iteration", fontsize=14*scaling)
    ax3.set_ylabel("Prediction error (MSE)", fontsize=14*scaling)
    ax3.tick_params(axis='x', labelsize=12*scaling)
    ax3.tick_params(axis='y', labelsize=12*scaling)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(max_n_locator))

    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()
    plt.close()

def plot_k_matrices(K_steady, K_opt, file_path, scaling=1.2, max_n_locator=5):
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)
    
    c1 = ax1.pcolormesh(K_steady, vmin=-0.1, vmax=0.45, cmap='RdBu_r')
    ax1.set_title('$K_\\mathrm{steady}$', fontsize=14*scaling)
    ax1.invert_yaxis()
    ax1.tick_params(axis='x', labelsize=12*scaling)
    ax1.tick_params(axis='y', labelsize=12*scaling)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(max_n_locator))
    fig.colorbar(c1,orientation='vertical')

    c2 = ax2.pcolormesh(K_opt, vmin=-0.1, vmax=0.45, cmap='RdBu_r')
    ax2.set_title('$K_\\mathrm{opt}$', fontsize=14*scaling)
    ax2.invert_yaxis()
    ax2.tick_params(axis='x', labelsize=12*scaling)
    ax2.tick_params(axis='y', labelsize=12*scaling)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(max_n_locator))

    #cb_ax = fig.add_axes([.93, .124, .02, .754])  # add and align colorbar
    fig.colorbar(c2,orientation='vertical')

    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()
    plt.close()








