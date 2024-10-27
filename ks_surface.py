import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
import argparse
import pickle
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run KS model with ensemble filters")
    parser.add_argument('--n_ensemble', type=int, default=20, help='Number of ensemble members')
    return parser.parse_args()

# Initialize parameters
N = 10 # number of Monte Carlo samples

num_steps = 1000  # Number of simulation steps
n = 256 # Dimensionality of the state space for KS model
observation_interval = 5  # Interval at which observations are made
dt = 0.25  # Time step for the KS model
J0 = 0
ks_model = KuramotoSivashinsky(dt=dt, s=n, l=22, M=16)
ks_step = Partial(ks_model.step)

@jit
def var_cost(radius, inflation, model, ensemble_init, observations, H, Q, R, key, J, J0):
    localization_matrix = generate_localization_matrix(n, radius)
    pred_states, pred_covariances, states, covariances = ensrf_steps(ks_step, ensemble_init, num_steps, observations, observation_interval, H, Q, R, localization_matrix, inflation, key)
    ensemble_mean = jnp.mean(states, axis=-1)  # Taking the mean across the ensemble members dimension
    ensemble_mean_pred = jnp.mean(pred_states, axis=-1)
    key, *subkeys = random.split(key, num=N+1)
    kl_sum = KL_sum(ensemble_mean_pred, pred_covariances, ensemble_mean, covariances, n, ks_step, Q, key)
    def inner_map(subkey):
        return log_likelihood(random.multivariate_normal(subkey, ensemble_mean, covariances), observations, H, R, J, J0)  
    cost = kl_sum - jnp.mean(jax.lax.map(inner_map, jnp.vstack(subkeys)))
    return cost

def main():
    args = parse_arguments()
    n_ensemble = args.n_ensemble    
    # Initial state
    key = random.PRNGKey(0)  # Random key for reproducibility
    x0 = random.normal(key, (n,))
    initial_state = x0
    
    # Noise covariances
    Q = 0.01 * jnp.eye(n)  # Process noise covariance
    R = 0.5 * jnp.eye(n)  # Observation noise covariance
    # Observation matrix (identity matrix for direct observation of all state variables)
    H = jnp.eye(n)
    
    # Generate observations
    observations, true_states = generate_true_states(key, num_steps, n, x0, H, Q, R, ks_step, observation_interval)
    
    
    # Run the simulation
    radius_range = jnp.arange(1, 100, 10)
    inflation_range = jnp.linspace(1.01, 1.9, 10)
    # radius_range = jnp.arange(1, 256, 200)
    # inflation_range = jnp.linspace(1.01, 1.9, 2)
    results = []
    
    # Initialize the ensemble
    ensemble_init = random.multivariate_normal(key, initial_state, Q, (n_ensemble,)).T
    cost_values = []
    
    # Loop through radius and inflation values
    for radius_opt in radius_range:
        # print(radius_opt)
        # print(datetime.now().strftime("%H:%M"))
        for infl in inflation_range:
            key, subkey = random.split(key)
            cost = var_cost(radius_opt, infl, ks_step, ensemble_init, observations, H, Q, R, subkey, num_steps, J0)
            cost_values.append((radius_opt, infl, cost))
    
    results.append(jnp.array(cost_values))
    
    # Save results to file
    with open('ks_ensemble' + str(n_ensemble) + '_data_range100.pkl', 'wb') as file:
        pickle.dump(results, file)
        
if __name__ == "__main__":
    main()

