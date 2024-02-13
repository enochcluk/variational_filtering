from jax.scipy.linalg import inv, det, svd
import jax.numpy as np
from jax import random, jit
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import jax


class BaseModel:
    def __init__(self, dt=0.01):
        self.dt = dt

    def step(self, x):
        raise NotImplementedError("The step method must be implemented by subclasses.")

class Lorenz63(BaseModel):
    def __init__(self, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3):
        super().__init__(dt)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    @jit
    def step(self, x):
        x_dot = self.sigma * (x[1] - x[0])
        y_dot = x[0] * (self.rho - x[2]) - x[1]
        z_dot = x[0] * x[1] - self.beta * x[2]
        return x + self.dt * np.array([x_dot, y_dot, z_dot])

class Lorenz96(BaseModel):
    def __init__(self, dt=0.01, F=8.0):
        super().__init__(dt)
        self.F = F

    @jit
    def step(self, x):
        def l96_step(x, F):
            return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, -1) - x + F
        return x + self.dt * l96_step(x, self.F)

class KuramotoSivashinsky(BaseModel):
    def __init__(self, dt=0.25, s=128, l=22, M=16):
        super().__init__(dt)
        self.s, self.l, self.M = s, l, M # discretization points, domain length, exponential time differenceing points (modes)
        self.k, self.E, self.E2, self.Q, self.f1, self.f2, self.f3, self.g = self.precompute_constants()

    def precompute_constants(self):
        k = (2 * np.pi / self.l) * np.concatenate([np.arange(0, self.s//2), np.array([0]), np.arange(-self.s//2+1, 0)])
        L = k**2 - k**4
        E = np.exp(self.dt*L)
        E2 = np.exp(self.dt*L/2)
        r = np.exp(1j * np.pi * (np.arange(1, self.M+1)-.5) / self.M)
        LR = self.dt * np.tile(L, (self.M, 1)).T + np.tile(r, (self.s, 1))
        Q = self.dt * np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
        f1 = self.dt * np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
        f2 = self.dt * np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
        f3 = self.dt * np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
        g = -0.5j * k
        return k, E, E2, Q, f1, f2, f3, g

    @jit
    def step(self, x):
        v = np.fft.fft(x)
        Nv = self.g * np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = self.E2 * v + self.Q * Nv
        Na = self.g * np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = self.E2 * v + self.Q * Na
        Nb = self.g * np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = self.E2 * a + self.Q * (2*Nb - Nv)
        Nc = self.g * np.fft.fft(np.real(np.fft.ifft(c))**2)
        v_next = self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
        x_next = np.real(np.fft.ifft(v_next))
        return x_next



def generate_true_states(key, num_steps, n, x0, C0, H, Q, R, model_step, observation_interval):
    # Initialize the state with the initial condition based on x0 and C0
    x0 = x0 + random.multivariate_normal(key, np.zeros(n), C0)
    x = np.zeros((num_steps, n))
    obs = np.zeros((num_steps, H.shape[0]))  # Adjust the shape based on H
    x = x.at[0].set(x0)

    for j in range(1, num_steps):
        key, subkey = random.split(key)
        # Update state using the model step function
        x_j = model_step(x[j-1])
        # Add process noise Q only at observation times
        if j % observation_interval == 0:
            x_j = x_j + random.multivariate_normal(subkey, np.zeros(n), Q)
            obs_state = np.dot(H, x_j)
            obs_noise = random.multivariate_normal(subkey, np.zeros(H.shape[0]), R)
            obs = obs.at[j].set(obs_state + obs_noise)
        else: #grabs previous observation - careful not to treat as current
            obs = obs.at[j].set(obs[j-1])
        
        x = x.at[j].set(x_j)

    return obs, x



def visualize_observations(observations):
    observation_values = observations.T  # Transpose for plotting

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list('CustomColormap', [(0, 'blue'), (0.5, 'white'), (1, 'red')])

    # Create a grid plot
    plt.figure(figsize=(12, 6))
    plt.imshow(observation_values, cmap=cmap, aspect='auto', interpolation='nearest', extent=[0, observations.shape[0], 0, observations.shape[1]])
    plt.colorbar(label='Observation Value')
    plt.xlabel('Time Step')
    plt.ylabel('State/Variable Number')
    plt.title('Observations Over Time')
    
  
    
    plt.show()

