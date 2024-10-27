# Import necessary libraries
import jax.numpy as jnp
from jax import random, grad, jit, lax, jacrev
from jax.scipy.linalg import inv, svd, eigh, det
from jax.numpy.linalg import norm
from tqdm.auto import tqdm
from sklearn.datasets import make_spd_matrix
from jax_models import visualize_observations, Lorenz96, KuramotoSivashinsky, generate_true_states, generate_localization_matrix
from jax_filters import ensrf_steps, kalman_filter_process, ensrf_step
import jax
import matplotlib.pyplot as plt
from jax.tree_util import Partial
from functools import partial
from jax_vi import KL_gaussian, log_likelihood, KL_sum
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_debug_nans", True)

N = 10 # number of Monte Carlo samples
n_ensemble = 5


num_steps = 10  # Number of simulation steps
observation_interval = 1  # Interval at which observations are made (THIS SHOULD WORK FOR NON-1 NUMBERS, REQUIRES FURTHER INVESTIGATION)
# n = 256  #KS params
# dt = 0.25

n = 40
dt = 0.05  # Time step for the L96 Model
J0 = 0
key = random.PRNGKey(0)  # Random key for reproducibility
x0 = random.normal(key, (n,))
initial_state = x0

# Noise covariances
Q = 0.01 * jnp.eye(n)  # Process noise covariance
R = 0.5 * jnp.eye(n)  # Observation noise covariance
# Observation matrix (identity matrix for direct observation of all state variables)
H = jnp.eye(n)
m0 = initial_state
C0 = Q


l96_model = Lorenz96(dt = dt, F = 8)
l96_step = Partial(l96_model.step)
# Generate true states and observations using the Lorenz '96 model

state_transition_function = l96_step


jacobian_function = jacrev(l96_step, argnums=0)
jac_func = Partial(jacobian_function)

observations, true_states = generate_true_states(key, num_steps, n, initial_state, H, Q, R, l96_step, observation_interval)

@partial(jit, static_argnums=(0))
def generate_distance_matrix(n, distances):
    """
    Generate a localization matrix with learned distances.
    
    distances: A 1D array of size n/2 representing the learned distances.
    """
    half_n = n // 2
    full_distances = jnp.concatenate([distances, distances[::-1]])  
    i = jnp.arange(n)[:, None]
    j = jnp.arange(n)
    min_modulo_distance = jnp.minimum(jnp.abs(i - j), n - jnp.abs(i - j))
    r = full_distances[min_modulo_distance.astype(int)] 
    localization_matrix = jnp.exp(-(r**2))
    return localization_matrix


@jit
def var_cost(distances, inflation, model, ensemble_init, observations, H, Q, R, key, J, J0):
    localization_matrix = generate_distance_matrix(n, distances)
    pred_states, pred_covar, states, covariances = ensrf_steps(state_transition_function, ensemble_init, num_steps, observations, observation_interval, H, Q, R, localization_matrix, inflation, key)
    ensemble_mean = jnp.mean(states, axis=-1)  # Taking the mean across the ensemble members dimension
    pred_mean = jnp.mean(pred_states, axis = -1)
    key, *subkeys = random.split(key, num=N+1)
    kl_sum = KL_sum(pred_mean, pred_covar, ensemble_mean, covariances, n, state_transition_function, Q, key)

    def inner_map(subkey):
        return log_likelihood(random.multivariate_normal(subkey, ensemble_mean, covariances), observations, H, R, num_steps, J0)  
    cost = kl_sum - jnp.nanmean(jax.lax.map(inner_map, jnp.vstack(subkeys)))
    
    return cost


inflation = 1.3  # Fixed starting value for inflation
alpha = 1e-4  # Learning rate
key = random.PRNGKey(0)  # Random key
N = 10  # Number of MC samples
m0 = initial_state
C0 = Q  # Initial covariance, assuming Q is your process noise covariance
ensemble_init = random.multivariate_normal(key, initial_state, Q, (n_ensemble,)).T

import properscoring
from IPython.display import clear_output
from jax import grad
from tqdm.notebook import tqdm
import jax.numpy as jnp
from jax import random
import pickle

# Initialize the distance array for gradient descent (n/2 values)
half_n = n // 2
key = random.PRNGKey(0)
distances_opt = jnp.cos(jnp.linspace(0, jnp.pi / 2, half_n))  # Cosine decay from 1 to 0
localization_matrix = generate_distance_matrix(n, distances_opt)


# Other parameters
inflation = 1.05  # Fixed starting value for inflation
alpha = 1e-1  # Learning rate
key = random.PRNGKey(0)  # Random key
N = 10  # Number of MC samples
m0 = initial_state
C0 = Q  # Initial covariance, assuming Q is your process noise covariance
ensemble_init = random.multivariate_normal(key, initial_state, Q, (n_ensemble,)).T

crpss = []
rmses = []
true_div = []
costs = []
covars = []
distances_list = []

# Run Classic Kalman
base_m, base_C, base_K  = kalman_filter_process(state_transition_function, jac_func, m0, C0, observations, H, Q, R)

var_cost_grad = grad(var_cost, argnums=0)

for i in tqdm(range(100)):
    key, subkey = random.split(key)
    
    current_cost = var_cost(distances_opt, inflation, state_transition_function, ensemble_init, observations, H, Q, R, subkey, num_steps, J0)
    costs.append(current_cost)
    grad_distances = var_cost_grad(distances_opt, inflation, state_transition_function, ensemble_init, observations, H, Q, R, subkey, num_steps, J0)
    
    distances_opt -= alpha * grad_distances
    distances_list.append(distances_opt)
    localization_matrix = generate_distance_matrix(n, distances_opt)

    # Run the ensemble filter steps with the updated localization matrix
    pred_states, pred_covar, states, covariances = ensrf_steps(state_transition_function, ensemble_init, num_steps, observations, observation_interval, H, Q, R, localization_matrix, inflation, key)
    
    ensemble_mean = jnp.mean(states, axis=-1)  # Taking the mean across the ensemble members dimension
    rmse = jnp.sqrt(jnp.mean((ensemble_mean - true_states)**2))
    rmses.append(rmse)
    crps = properscoring.crps_ensemble(true_states, states).mean(axis=1).mean()
    crpss.append(crps)
    total_kl_divergence = 0
    for t in range(num_steps):  
        kl_div_t = KL_gaussian(n, ensemble_mean[t], covariances[t],  base_m[t], base_C[t])
        total_kl_divergence += kl_div_t
    true_div.append(total_kl_divergence / num_steps)
    covars.append(covariances)
    
    print(f"Iteration {i+1}, Distance Gradient Norm: {jnp.linalg.norm(grad_distances)}, RMSE: {rmse}")

results = {
    'distances': distances_list,
    'crpss': crpss,
    'rmses': rmses,
    'true_div': true_div,
    'costs': costs,
}
from datetime import datetime

current_time = datetime.now().strftime("%H-%M")


with open(f'l96_results_with_cos_distances_{current_time}.pkl', 'wb') as f:
    pickle.dump(results, f)
