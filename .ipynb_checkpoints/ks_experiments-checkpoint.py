import importlib
import jax_models
from jax.scipy.linalg import inv, det, svd
import jax.numpy as np
from jax import random, jit
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import jax
from jax_models import visualize_observations, KuramotoSivashinsky, generate_true_states, generate_gc_localization_matrix
from jax_filters import ensrf_steps
importlib.reload(jax_models)
# Initialize parameters
num_steps = 1000  # Number of simulation steps
n = 256 # Dimensionality of the state space for KS model
observation_interval = 5  # Interval at which observations are made
dt = 0.25  # Time step for the KS model

ks_model = KuramotoSivashinsky(dt=dt, s=n, l=22, M=16)

import pickle
import os
from datetime import datetime
from tqdm import tqdm

radii = [2,5, 10, 50, 100]
inflations = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
num_trials = 10
ensemble_sizes = [5, 10,20, 30,40]

n = ks_model.s
num_steps = 1000  # Total number of time steps
observation_interval = 5  # Observation is available every 5 time steps
observation_times = np.arange(0, num_steps, observation_interval)

# Initialize random key for reproducibility
key = random.PRNGKey(0)

# Process noise covariance and Observation noise covariance
Q = 0.1 * np.eye(n)
R = 0.5 * np.eye(n)
# Observation matrix (identity matrix for direct observation of all state variables)
H = np.eye(n)

# Initial state
initial_state = random.normal(key, (n,))

# Initialize data structures for results
std_errors = {(radius, inflation, n_ensemble): [] for radius in radii for inflation in inflations for n_ensemble in ensemble_sizes}
errors = {(radius, inflation, n_ensemble): [] for radius in radii for inflation in inflations for n_ensemble in ensemble_sizes}

for trial in tqdm(range(num_trials), desc="Running Trials"):
    print(trial)
    observations, true_states = generate_true_states(key, num_steps, n, initial_state, H, Q, R, ks_model.step, observation_interval)
    observed_true_states = true_states[observation_times] #only comparing metrics for analysis steps
    for radius in radii:
        local_mat = generate_gc_localization_matrix(n, radius)
        for inflation in inflations:
            for n_ensemble in ensemble_sizes:
                print(n_ensemble)
                ensemble_init = random.multivariate_normal(key, initial_state, Q, (n_ensemble,)).T
                states = ensrf_steps(ks_model.step, n_ensemble, ensemble_init, num_steps, observations, observation_interval, H, Q, R, local_mat, inflation)
                average_state = np.mean(states, axis=2)  # Calculate the mean along the ensemble dimension
                observed_average_state = average_state[observation_times] #only comparing metrics for analysis steps
                error = np.sqrt(np.mean((observed_average_state - observed_true_states) ** 2, axis=1))
                errors[(radius, inflation, n_ensemble)].append(error)
                # Select states at observation times for std deviation calculation and calculate std deviation
                observed_states = states[:, observation_times, :]
                std_dev = np.mean(np.std(observed_states, axis = 2))  # Standard deviation across all ensemble members and state dimensions at observation times
                std_errors[(radius, inflation, n_ensemble)].append(std_dev)
                
                # error = np.sqrt(np.mean((average_state - true_states) ** 2, axis=1))
                # errors[(radius, inflation, n_ensemble)].append(error)
                # std_dev = np.mean(np.std(states, axis = 2))  # Standard deviation across all ensemble members and state dimensions
                # std_errors[(radius, inflation, n_ensemble)].append(std_dev)

# Preparing data for saving
all_data = {
    'std_errors': std_errors,
    'errors': errors,
    'parameters': {
        'radii': radii,
        'inflations': inflations,
        'ensemble_sizes': ensemble_sizes,
        'num_trials': num_trials,
        'filter_params': (H,Q,R, observation_interval, num_steps)
    }
}

# File saving path
directory = '/central/home/eluk/variational_filtering/experiment_data/'
filename = 'feb22.pkl'
file_path = os.path.join(directory, filename)

with open(file_path, 'wb') as f:
    pickle.dump(all_data, f)