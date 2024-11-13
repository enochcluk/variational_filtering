# Learning Optimal Filters Using Variational Inference

This repository contains the code for the paper ["Learning Optimal Filters Using Variational Inference"](https://arxiv.org/abs/2406.18066) presented at ICML 2024 Workshop "ML4ESM". The project implements methods for learning parameterized filters using variational inference, particularly for dynamical systems like the Lorenz-96 model. The goal is to estimate gain matrices and tune parameters like inflation and localization in Ensemble Kalman Filters (EnKFs).

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Experiments](#experiments)
5. [References](#references)

## Introduction

Filtering is the task of estimating the states of a dynamical system given partial, noisy observations. While closed-form solutions like the Kalman filter exist for linear systems, high-dimensional, nonlinear systems require approximations. In this project, we present a method for learning the analysis map — the function that computes the filtering distribution from the forecast distribution and the observations — using variational inference.

We show how this approach can be used to:
- Learn gain matrices for filtering in both linear and nonlinear systems.
- Optimize inflation and localization parameters in an EnKF.

Our method leverages automatic differentiation in JAX for fast and efficient optimization.

## Installation

Clone the repository. 

Ensure you have JAX installed for GPU acceleration (and other dependencies):

```bash
pip install -r requirements.txt
```

## Usage

### Running the Code

To run the filtering experiments, you can run all of the cells of any jupyter notebook in sequence.
It may be faster to create data with batched python scripts and only use jupyter for visualization.

### Key Modules

- **jax_models.py**: Contains implementations of various dynamical systems such as the Lorenz-96 model and the Kuramoto-Sivashinsky equation.
- **jax_filters.py**: Implements filtering algorithms like the Ensemble Kalman Filter (EnKF) and Kalman filter with fixed and learned gains.
- **jax_vi.py**: Contains utility functions for variational inference, including KL divergence calculations and log-likelihood estimation.


## Experiments

We performed various experiments to test the effectiveness of the approach, including:

1. **Gain Learning for Linear and Nonlinear Systems**:
   - We learn a constant gain matrix in linear and nonlinear systems, including the Lorenz-96 model.
   - The results are compared to the optimal steady-state gain of a Kalman filter in linear cases and the extended Kalman filter (ExKF) for nonlinear models.
   - We show both an online (learn with each new observation) and offline (use all timesteps to create loss) approach

2. **Ensemble Kalman Filter (EnKF) with Learned Inflation and Localization**:
   - We tune the inflation and localization parameters for EnKF using variational inference to match the true filter as closely as possible.
   - We also generate loss surfaces for visual inspection, as this is a 2D problem
  
   
### Visualizing Results

Several plotting functions are available to visualize different steps of the filtering process:

- **`visualize_observations()`**: Visualizes observations over time.
- **`plot_ensemble_mean_and_variance()`**: Plots the ensemble mean and variance with the 95% confidence interval.
- **`plot_optimization_results()`**: Plots the optimization results, including gain errors and prediction error metrics.
- **`plot_k_matrices()`**: Visualizes the comparison of Kalman gains, showing the differences between the learned gain and the steady-state Kalman gain.

The first two functions are found in jax_models, while the latter two are found in jax_vi.

## References

- Boudier, P., Fillion, A., Gratton, S., G{\"u}rol, S., & Zhang, S. (2023). Data Assimilation Networks. *Journal of Advances in Modeling Earth Systems*, 15(4), e2022MS003353. https://doi.org/10.1029/2022MS003353
- Hoang, H.S., De Mey, P., & Talagrand, O. (1994). A Simple Adaptive Algorithm of Stochastic Approximation Type for System Parameter and State Estimation. *Proceedings of 1994 33rd IEEE Conference on Decision and Control*, 1, 747-752. https://doi.org/10.1109/CDC.1994.410863
- Reich, S., & Cotter, C. (2015). *Probabilistic forecasting and Bayesian data assimilation*. Cambridge University Press.
- Hoang, S., Baraille, R., Talagrand, O., Carton, X., & De Mey, P. (1998). Adaptive Filtering: Application to Satellite Data Assimilation in Oceanography. *Dynamics of Atmospheres and Oceans*, 27(1), 257-281. https://doi.org/10.1016/S0377-0265(97)00014-6
- Lambert, M., Bonnabel, S., & Bach, F. (2023). Variational Gaussian Approximation of the Kushner Optimal Filter. In F. Nielsen & F. Barbaresco (Eds.), *Geometric Science of Information* (pp. 395-404). Springer Nature Switzerland. https://doi.org/10.1007/978-3-031-38271-0_39
- Levine, M., & Stuart, A. (2022). A Framework for Machine Learning of Model Error in Dynamical Systems. *Communications of the American Mathematical Society*, 2(07), 283-344. https://doi.org/10.1090/cams/10
- McCabe, M., & Brown, J. (2021). Learning to Assimilate in Chaotic Dynamical Systems. *Advances in Neural Information Processing Systems*, 34, 12237-12250.
- Marino, J., Cvitkovic, M., & Yue, Y. (2018). A General Method for Amortizing Variational Filtering. *Advances in Neural Information Processing Systems*, 31.
- Snyder, C., Bengtsson, T., Bickel, P., & Anderson, J. (2008). Obstacles to high-dimensional particle filtering. *Monthly Weather Review*, 136(12), 4629-4640.
- Sanz-Alonso, D., Stuart, A., & Taeb, A. (2023). *Inverse Problems and Data Assimilation*. Cambridge University Press. https://doi.org/10.1017/9781009414319
