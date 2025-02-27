import jax
import jax.numpy as jnp
from jax import random, grad, jit, jacfwd, jacrev, value_and_grad, lax
from jax.scipy.linalg import inv, svd, eigh, det
from jax.lax import scan
from scipy.linalg import solve_discrete_are, norm

import pickle
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from jax.tree_util import Partial
from functools import partial
from jax_vi import KL_gaussian, log_likelihood, KL_sum, plot_optimization_results, plot_k_matrices
from jax_filters import apply_filtering_fixed_nonlinear, kalman_filter_process, ensrf_steps
from jax_models import visualize_observations, Lorenz63, generate_true_states, generate_localization_matrix

from flax.training import train_state
from functools import partial
import optax
import argparse

import scoringrules as sr
import properscoring as ps

jax.config.update("jax_platform_name", "gpu")

# Argument Parser
parser = argparse.ArgumentParser(description="Train model and save results.")
parser.add_argument("--output", type=str, default="losses.pkl", help="Filename to save the losses.")
args = parser.parse_args()
print("Experiment", args.output)

N = 10  # number of Monte Carlo samples
num_steps = 10000  # Total number of time steps
num_train_steps = 7000  # Number of training time steps
num_test_steps = num_steps - num_train_steps  # Number of testing time steps
J0 = 0
n = 3   # Number of state variables
key = random.PRNGKey(42)  # Random key for reproducibility
Q = 0.1 * jnp.eye(n)  # Process noise covariance
R = 0.05 * jnp.eye(n)  # Observation noise covariance
H = jnp.eye(n)  # Observation matrix (identity matrix for direct observation of all state variables)

n_ensemble = 10
observation_interval = 1
initial_state = random.normal(random.PRNGKey(42), (n,)) 
m0 = initial_state
C0 = Q
ensemble_init = random.normal(key, (n, n_ensemble))  # Use a specific subkey

l63_model = Lorenz63()
l63_step = Partial(l63_model.step)

jacobian_function = jacrev(l63_step, argnums=0)
jac_func = Partial(jacobian_function)
state_transition_function = l63_step

# Generate true states and observations for 1000 steps
true_states, observations = generate_true_states(key, num_steps, n, initial_state, H, Q, R, l63_step, observation_interval)
train_observations = observations[:num_train_steps]

@partial(jit, static_argnums=(2, 5))
def nn_analysis_filter_steps(
    state_transition_function,
    ensemble_init,
    num_steps,
    observations,
    observation_interval,
    model,
    params,
    key,
):
    model_vmap = jax.vmap(lambda v: state_transition_function(v), in_axes=1, out_axes=1)
    key, *subkeys = random.split(key, num=num_steps + 1)
    subkeys = jnp.array(subkeys)

    def inner(carry, t):
        ensemble = carry
        ensemble_predicted = model_vmap(ensemble)
        def true_fun(_):
            # Flatten the predicted ensemble and prepare input for the model
            pred_flat = ensemble_predicted.reshape(-1)  # (n_ensemble * n,)
            input_t = jnp.concatenate([pred_flat, observations[t]])  # Append observation
            # Use the NN to predict the analysis ensemble
            analysis_flat = model.apply(params, input_t)
            # Reshape back to (n, n_ensemble)
            analysis_ensemble = analysis_flat.reshape(ensemble_predicted.shape)
            return analysis_ensemble
            

        def false_fun(_):
            return ensemble_predicted

        updated_ensemble = lax.cond(
            t % observation_interval == 0, true_fun, false_fun, operand=None
        )
        return updated_ensemble, (ensemble_predicted, updated_ensemble)

    # Perform filtering over all time steps
    _, (ensemble_preds, ensembles) = lax.scan(
        inner, ensemble_init, jnp.arange(num_steps)
    )

    return ensemble_preds, ensembles
    
@jit
def energy_score(obs, fct, eps=1e-8):
    #  based on scoringrules library
    
    M = fct.shape[-2]  # Get ensemble size (M)
    obs = jnp.expand_dims(obs, axis=-2)  # Expand obs to match ensemble

    # Safe norm computation
    def safe_norm(x):
        return jnp.sqrt(jnp.sum(x**2, axis=-1) + eps)

    e_1 = jnp.sum(safe_norm(fct - obs)) / M
    e_2 = jnp.sum(safe_norm(fct[:, :, None, :] - fct[:, None, :, :])) / (M**2)

    return e_1 - 0.5 * e_2
    
    
from flax import linen as nn

class AnalysisNet(nn.Module):
    input_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x
        
key, subkey = random.split(key)
kl_model = AnalysisNet(input_dim=(n_ensemble + 1) * n, output_dim=n_ensemble * n)
dummy_input = jnp.zeros((1, (n_ensemble + 1) * n)) 
params = kl_model.init(subkey, dummy_input)#params = mse_state.params


train_observations = observations[:num_train_steps]

@jit
def var_cost(params):
    key = random.PRNGKey(42)  
    key, *subkeys = random.split(key, num=N+1)  

 
    ensemble_init = random.normal(subkeys[0], (n, n_ensemble))  # Use a specific subkey

    ensemble_preds, ensembles = nn_analysis_filter_steps(
        state_transition_function,
        ensemble_init,
        num_train_steps,
        train_observations,
        observation_interval,
        kl_model,
        params,
        subkeys[1]  # Pass a different subkey for randomness
    )
    analysis_ensemble_default = jnp.swapaxes(ensembles, 1, 2)
    energy_scores = energy_score(true_states[:num_train_steps], analysis_ensemble_default)
    loss = jnp.sum(energy_scores)
    
    return loss


warmup_steps = 10  # Number of warmup steps
initial_lr = 5e-4
min_lr_factor = 0.5  # Final learning rate as a fraction of initial_lr

schedule = optax.join_schedules(
    schedules=[
        optax.linear_schedule(init_value=initial_lr, end_value=initial_lr, transition_steps=warmup_steps),
        optax.cosine_decay_schedule(init_value=initial_lr, decay_steps=num_train_steps, alpha=min_lr_factor)
    ],
    boundaries=[warmup_steps]
)

# Optimizer with dynamic learning rate
tx = optax.adam(schedule)
kl_state = train_state.TrainState.create(apply_fn=kl_model.apply, params=params, tx=tx)

@jit
def train_step(state):
    loss, grads = value_and_grad(var_cost)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop
num_epochs = 500

losses = []

for epoch in tqdm(range(num_epochs)):    
    kl_state, loss = train_step(kl_state)
    losses.append(loss)
    if jnp.isnan(loss):
        break
    if epoch % 5 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.2e}")

# MSE Loss Calculation
def compute_mse_for_all_timesteps(pred_means, true_states):
    mse_vals = jnp.mean((pred_means - true_states) ** 2, axis=1)  # Shape: (num_steps,)
    return jnp.mean(mse_vals)

# Neural Network Filter
ensemble_preds, nn_ensemble = nn_analysis_filter_steps(
    state_transition_function=l63_step,
    ensemble_init=ensemble_init,
    num_steps=num_steps,
    observations=observations,
    observation_interval=1,
    model=kl_model,
    params=kl_state.params,
    key=key,
)

# Slice from num_train_steps onwards
ensemble_means = jnp.mean(nn_ensemble, axis=2)[num_train_steps:]  # Shape: (num_steps - num_train_steps, n)
pred_means = jnp.mean(ensemble_preds, axis=2)[num_train_steps:]
true_states_eval = true_states[num_train_steps:]

# Compute CRPS using properscoring for the neural network-based filtering
crps_nn_analysis = jnp.mean(jnp.array([ps.crps_ensemble(true_states_eval[t], nn_ensemble[t + num_train_steps]) 
                                       for t in range(nn_ensemble.shape[0] - num_train_steps)]))
crps_nn_pred = jnp.mean(jnp.array([ps.crps_ensemble(true_states_eval[t], ensemble_preds[t + num_train_steps]) 
                                   for t in range(ensemble_preds.shape[0] - num_train_steps)]))

# Compute MSE for the neural network-based filtering
mse_nn_analysis = compute_mse_for_all_timesteps(ensemble_means, true_states_eval)
mse_nn_pred = compute_mse_for_all_timesteps(pred_means, true_states_eval)

print("CRPS Analysis to True State (NN)", crps_nn_analysis)
#print("CRPS Predicted to True State (NN)", crps_nn_pred)
print("MSE Analysis to True State (NN)", mse_nn_analysis)
#print("MSE Predicted to True State (NN)", mse_nn_pred)




@jit
def kalman_step(state, observation, params):
    m_prev, C_prev = state
    state_transition_function, jacobian_function, H, Q, R = params
    
    # Prediction step
    m_pred = state_transition_function(m_prev)
    F_jac = jacobian_function(m_prev)
    C_pred = F_jac @ C_prev @ F_jac.T + Q
    
    # Update step
    S = H @ C_pred @ H.T + R
    K_curr = C_pred @ H.T @ jnp.linalg.inv(S)
    m_update = m_pred + K_curr @ (observation - H @ m_pred)
    C_update = (jnp.eye(H.shape[1]) - K_curr @ H) @ C_pred
    
    return (m_update, C_update), (m_pred, C_pred, m_update, C_update, K_curr)

@jit
def kalman_filter_process(state_transition_function, jacobian_function, m0, C0, observations, H, Q, R):
    params = (state_transition_function, jacobian_function, H, Q, R)
    initial_state = (m0, C0)
    
    # Modified scan to capture both prediction and analysis states
    _, (m_preds, C_preds, m_updates, C_updates, Ks) = lax.scan(
        lambda state, obs: kalman_step(state, obs, params),
        initial_state, 
        observations
    )
    
    return m_preds, C_preds, m_updates, C_updates, Ks


m_preds, C_preds, m_updates, C_updates, Ks = kalman_filter_process(state_transition_function, jac_func, m0, C0, observations, H, Q, R)

key, subkey = random.split(key)
ensemble_init = random.multivariate_normal(subkey, m0, C0, (n_ensemble,)).T  # Shape: (n, ensemble_size)
localization_matrix = generate_localization_matrix(3,1)

ensemble_preds, C_preds, ensembles, covariances = ensrf_steps(state_transition_function, ensemble_init, num_train_steps, observations, 1, H, Q, R, localization_matrix=localization_matrix, inflation=1.9, key=key)




# Compute CRPS for extended Kalman Filter (EKF) using properscoring
crps_ekf = jnp.mean(jnp.array([ps.crps_ensemble(true_states_eval[t], m_updates[t + num_train_steps]) 
                               for t in range(m_updates.shape[0] - num_train_steps)]))
mse_ekf = compute_mse_for_all_timesteps(m_updates[num_train_steps:], true_states_eval)

print("CRPS Loss from Kalman Filter", crps_ekf)
print("MSE Loss from Kalman Filter", mse_ekf)

# Ensemble Kalman Filter (EnsRF)
key, subkey = random.split(key)
ensemble_init = random.multivariate_normal(subkey, m0, C0, (n_ensemble,)).T  # Shape: (n, ensemble_size)
localization_matrix = generate_localization_matrix(3, 1)

ensemble_preds, C_preds, ensembles, covariances = ensrf_steps(
    state_transition_function, ensemble_init, num_steps, observations, 
    1, H, Q, R, localization_matrix=localization_matrix, inflation=1.9, key=key
)

# Compute CRPS for the ensemble-based filtering method (EnsRF) using properscoring
crps_ensrf = jnp.mean(jnp.array([ps.crps_ensemble(true_states_eval[t], ensembles[t + num_train_steps]) 
                                 for t in range(ensembles.shape[0] - num_train_steps)]))
mse_ensrf = compute_mse_for_all_timesteps(jnp.mean(ensembles, axis=2)[num_train_steps:], true_states_eval)

print("CRPS from Ensemble Kalman Filter (EnsRF)", crps_ensrf)
print("MSE from Ensemble Kalman Filter (EnsRF)", mse_ensrf)

with open("args.output", "wb") as f:
    pickle.dump({
        "mse_nn": mse_nn_analysis,
        "crps_nn": crps_nn_analysis,
        "mse_kalman": mse_ekf,
        "crps_kalman": crps_ekf,
        "mse_ensrf": mse_ensrf,
        "crps_ensrf": crps_ensrf,
        "training_losses": losses
    }, f)
    


