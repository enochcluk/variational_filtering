import jax.numpy as jnp
from jax import random, grad, jit, jacrev
from jax.scipy.linalg import inv, svd, eigh, det
from jax.numpy.linalg import norm
from tqdm.auto import tqdm
from sklearn.datasets import make_spd_matrix
from jax_models import visualize_observations, Lorenz96, KuramotoSivashinsky, generate_true_states, generate_localization_matrix
from jax_filters import ensrf_steps, kalman_filter_process
import jax
import matplotlib.pyplot as plt
from jax.tree_util import Partial
from functools import partial
from jax_vi import KL_gaussian, log_likelihood, KL_sum
import argparse
import pickle
from datetime import datetime
import properscoring


# Set JAX configuration
jax.config.update("jax_enable_x64", True)

# Argument parsing
parser = argparse.ArgumentParser(description="Data assimilation framework")
parser.add_argument('--model', type=str, choices=['l96', 'ks'], required=True, help="Model to use: 'l96' or 'ks'")
parser.add_argument('--num_steps', type=int, required=True, help="Number of simulation steps")
parser.add_argument('--initialization', type=str, choices=['sin', 'cos', 'random'], required=True, help="Initialization method: 'sin', 'cos', or 'random'")
parser.add_argument('--n_ensemble', type=int, required=True, help="Number of ensemble members")

args = parser.parse_args()

# Set parameters based on command-line arguments
model_type = args.model
num_steps = args.num_steps
initialization_method = args.initialization
n_ensemble = args.n_ensemble

N = 10
inflation = 1.3
alpha = 1e-2

# Initialize the model based on the command-line argument
if model_type == 'ks':
    n = 256  # default value for KS, changeable based on the model
    dt = 0.25
    ks_model = KuramotoSivashinsky(dt=dt, s=n, l=22, M=16)
    state_transition_function = Partial(ks_model.step)
elif model_type == 'l96':
    n = 40  # default for Lorenz '96
    dt = 0.05
    l96_model = Lorenz96(dt=dt, F=8)
    state_transition_function = Partial(l96_model.step)

jacobian_function = jacrev(state_transition_function, argnums=0)
jac_func = Partial(jacobian_function)

# Initialize the state
key = random.PRNGKey(0)
x0 = random.normal(key, (n,))

initial_state = x0

# Noise covariances
Q = 0.01 * jnp.eye(n)  # Process noise covariance
R = 0.05 * jnp.eye(n)  # Observation noise covariance
H = jnp.eye(n)  # Observation matrix (identity matrix for direct observation of all state variables)
m0 = initial_state
C0 = Q

# Generate observations
observations, true_states = generate_true_states(key, num_steps, n, initial_state, H, Q, R, state_transition_function, observation_interval=1)

@partial(jit, static_argnums=(0))
def generate_distance_matrix(n, distances):
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
    pred_states, pred_covar, states, covariances = ensrf_steps(state_transition_function, ensemble_init, num_steps, observations, 1, H, Q, R, localization_matrix, inflation, key)
    ensemble_mean = jnp.mean(states, axis=-1)
    pred_mean = jnp.mean(pred_states, axis=-1)
    key, *subkeys = random.split(key, num=N+1)
    kl_sum = KL_sum(pred_mean, pred_covar, ensemble_mean, covariances, n, state_transition_function, Q, key)

    def inner_map(subkey):
        return log_likelihood(random.multivariate_normal(subkey, ensemble_mean, covariances), observations, H, R, num_steps, J0)  
    cost = kl_sum - jnp.nanmean(jax.lax.map(inner_map, jnp.vstack(subkeys)))
    
    return cost

# Initialization for distance array
half_n = n // 2


if initialization_method == 'sin':
    distances_opt = jnp.cos(jnp.linspace(0, jnp.pi / 2, half_n))
elif initialization_method == 'cos':
    distances_opt = jnp.sin(jnp.linspace(0, jnp.pi / 2, half_n))
elif initialization_method == 'random':
    distances_opt = random.normal(key, (half_n,))

# Other parameters
ensemble_init = random.multivariate_normal(key, initial_state, Q, (n_ensemble,)).T

crpss = []
rmses = []
true_div = []
costs = []
distances_list = []

# Run Classic Kalman
base_m, base_C, base_K = kalman_filter_process(state_transition_function, jac_func, m0, C0, observations, H, Q, R)

var_cost_grad = grad(var_cost, argnums=0)

for i in tqdm(range(100)):
    key, subkey = random.split(key)
    
    current_cost = var_cost(distances_opt, inflation, state_transition_function, ensemble_init, observations, H, Q, R, subkey, num_steps, J0=0)
    costs.append(current_cost)
    grad_distances = var_cost_grad(distances_opt, inflation, state_transition_function, ensemble_init, observations, H, Q, R, subkey, num_steps, J0=0)
    
    distances_opt -= alpha * grad_distances
    distances_list.append(distances_opt)
    localization_matrix = generate_distance_matrix(n, distances_opt)

    pred_states, pred_covar, states, covariances = ensrf_steps(state_transition_function, ensemble_init, num_steps, observations, 1, H, Q, R, localization_matrix, inflation, key)
    
    ensemble_mean = jnp.mean(states, axis=-1)
    rmse = jnp.sqrt(jnp.mean((ensemble_mean - true_states)**2))
    rmses.append(rmse)
    crps = properscoring.crps_ensemble(true_states, states).mean(axis=1).mean()
    crpss.append(crps)
    total_kl_divergence = 0
    for t in range(num_steps):  
        kl_div_t = KL_gaussian(n, ensemble_mean[t], covariances[t], base_m[t], base_C[t])
        total_kl_divergence += kl_div_t
    true_div.append(total_kl_divergence / num_steps)

    print(f"Iteration {i+1}, Distance Gradient Norm: {jnp.linalg.norm(grad_distances)}, RMSE: {rmse}")

results = {
    'distances': distances_list,
    'crpss': crpss,
    'rmses': rmses,
    'true_div': true_div,
    'costs': costs,
}

current_time = datetime.now().strftime("%H-%M")
save_filename = f'{model_type}_results_{initialization_method}_steps{num_steps}_ensemble{n_ensemble}_{current_time}.pkl'

with open(save_filename, 'wb') as f:
    pickle.dump(results, f)
