# Bayesian Nonlinear Pricing Models with Regime-Switching and Adaptive Volatility Filters

## Overview

This repository provides a complete Python implementation of a Bayesian framework for estimating nonlinear asset pricing models that incorporate regime-switching and stochastic volatility. The model integrates a Hidden Markov Model (HMM) for discrete regime states, nonlinear stochastic volatility dynamics, and a Particle Markov Chain Monte Carlo (PMCMC) algorithm for joint state-parameter inference. We employ an Unscented Kalman Filter (UKF)-based filtering procedure to adaptively estimate latent volatility under uncertainty in both parameters and regimes.

The provided code offers:

- **Nonlinear State-Space Representation:**  
  Handles complex observation models relating latent volatility and regimes to observed prices.
  
- **Regime-Switching Dynamics (HMM):**  
  Integrates discrete state transitions to capture shifts in market conditions, sentiment, or liquidity.
  
- **Stochastic Volatility Modeling:**  
  Employs a mean-reverting stochastic volatility process (Heston-type dynamics) conditioned on regimes.
  
- **Unscented Kalman Filter (UKF):**  
  Efficient approximation of nonlinear filtering distributions for the continuous state (volatility).
  
- **Particle MCMC (PMCMC) for Parameter Estimation:**  
  Joint inference over model parameters and latent states via a combination of particle filtering and MCMC sampling steps.

This project is intended for quantitative researchers and practitioners interested in advanced Bayesian methods for financial time series and state-space modeling.

---

## Mathematical Foundations

### 1. Model Setup

We consider a discrete-time asset price series $$(S_t)_{t=1}^T $$. To model price dynamics under uncertainty and structural changes, we introduce:

1. **Regime Variable** \( R_t \):  
   A latent Markov chain representing market regimes:
   
$$
   R_t \in \{1,2,\dots,M\}, \quad P(R_t=j \mid R_{t-1}=i) = \Pi_{ij}.
$$

   The regime influences the parameters governing the price and volatility evolution.

3. **Stochastic Volatility State** \( v_t \):  
   Let \( v_t > 0 \) denote the latent volatility level at time \( t \). We assume a mean-reverting square-root process (Heston-type):
   
$$
   v_t = v_{t-1} + \kappa_{v,R_t}(\theta_{v,R_t} - v_{t-1}) + \sigma_{v,R_t}\sqrt{\max(v_{t-1}, 0)}\,\eta_t, \quad \eta_t \sim N(0,1).
$$

5. **Observation Model (Prices)**:
   We consider log-returns:
   
$$
   r_t = \log\left(\frac{S_t}{S_{t-1}}\right).
$$
   In regime \( R_t = i \), we have:
   
$$
   r_t \mid v_t, R_t=i \sim N(\mu_i, v_t\sigma_{level,i}^2).
$$

   Thus:
   
$$
   \log(S_t) = \log(S_{t-1}) + r_t, \quad r_t \sim N(\mu_i, v_t \sigma_{level,i}^2).
$$
   
   The parameters \( \{\mu_i, \sigma_{level,i}, \kappa_{v,i}, \theta_{v,i}, \sigma_{v,i}\} \) differ across regimes \( i \).

### 2. Hidden Markov Model (HMM)

The regime process \( (R_t) \) is a first-order Markov chain with transition matrix \( \Pi \). Let:

$$
\Pi = [\Pi_{ij}]_{i,j=1}^M, \quad \Pi_{ij} = P(R_t=j \mid R_{t-1}=i).
$$

The joint distribution of \( \{R_t\} \) given \( \Pi \) is:

$$
p(R_1,\ldots,R_T \mid \Pi) = \pi_{R_1} \prod_{t=2}^T \Pi_{R_{t-1},R_t},
$$

where \( \pi \) is the initial regime distribution (often stationary).

### 3. State-Space Formulation

Define the continuous latent state \( \mathbf{x}_t = v_t \). The joint model is:

1. **State Transition:**
   
$$
   v_t \mid v_{t-1}, R_t \sim N\left(v_{t-1} + \kappa_{v,R_t}(\theta_{v,R_t} - v_{t-1}), \sigma_{v,R_t}^2 v_{t-1}\right) \quad \text{(with non-negativity enforced)}
$$

2. **Observation Model:**

$$
   \log(S_t) \mid v_t, R_t=i \sim N(\log(S_{t-1}) + \mu_i, v_t \sigma_{level,i}^2).
$$

This is a nonlinear state-space model with discrete switching (HMM). The nonlinearity arises from the volatility dynamics and lognormal price observation.

### 4. Bayesian Hierarchical Structure

We place priors on model parameters. For each regime \( i \):

- \( \kappa_{v,i} \) may have a Gamma prior: \( \kappa_{v,i} \sim \Gamma(a_\kappa,b_\kappa) \).
- \( \theta_{v,i} \) may have a Gaussian prior: \( \theta_{v,i} \sim N(\mu_\theta,\sigma_\theta^2) \).
- \( \sigma_{v,i} \) may have a Half-Cauchy prior to ensure positivity and heavy-tailed flexibility.
- \( \mu_i \sim N(0,\sigma_\mu^2) \), and \( \sigma_{level,i} \) also from a Half-Cauchy.
  
For the regime transition matrix \( \Pi \), each row can have a Dirichlet prior to ensure valid probability distributions:
$$
\Pi_{i,:} \sim \text{Dirichlet}(\alpha,\alpha,\dots,\alpha).
$$

The resulting joint posterior is:
$$
p(\Theta, \{v_t\}, \{R_t\} \mid \{S_t\}_{t=1}^T) \propto p(\{S_t\} \mid \{v_t\},\{R_t\},\Theta)p(\{v_t\},\{R_t\}\mid\Theta)p(\Theta).
$$

### 5. Nonlinear Filtering: The Unscented Kalman Filter (UKF)

Given known parameters \( \Theta \), we seek the filtering distribution:
$$
p(v_t, R_t \mid S_{1:t}, \Theta).
$$

Since \( \{R_t\} \) is discrete, we can compute regime probabilities and state distributions simultaneously. The UKF provides a Gaussian approximation for the conditional distribution of \( v_t \) given \( R_t=i \). For each regime, we maintain a Gaussian approximation of the volatility state distribution. The overall distribution is a mixture:
$$
p(v_t \mid S_{1:t}, \Theta) = \sum_{i=1}^M p(R_t=i \mid S_{1:t}, \Theta) \, p(v_t \mid R_t=i, S_{1:t}, \Theta).
$$

**UKF Steps per Regime:**

1. **Sigma Points:** Generate a set of sigma points \( \{\chi^{(j)}_{t-1}\} \) from the Gaussian distribution at time \( t-1 \).
  
2. **Time Update:** Propagate each sigma point through the nonlinear state equation:
   $$
   \chi^{(j)}_t = f_{R_t}(\chi^{(j)}_{t-1}).
   $$
   Compute the predicted mean and covariance.
  
3. **Measurement Update:** Propagate sigma points through the observation model:
   $$
   y^{(j)}_t = h_{R_t}(\chi^{(j)}_t),
   $$
   update state mean and covariance using the observed \( S_t \).
  
Since \( R_t \) is unknown, we form a mixture by weighting each regime's UKF prediction by the HMM probabilities and integrate out the regimes.

---

## Particle MCMC (PMCMC) for Joint Parameter and State Inference

With uncertain parameters \( \Theta \), we must integrate over them as well. Direct inference is challenging due to the nonlinear, non-Gaussian nature of the model.

**Approach: Particle MCMC (Andrieu et al., 2010)**

1. **Particle Filter for Latent States:**
   Given a candidate parameter set \( \Theta \), we run a particle filter to approximate:
   $$
   \hat{p}(S_{1:T} \mid \Theta) \approx p(S_{1:T} \mid \Theta).
   $$
   The particle filter simulates \( (v_t^{(n)}, R_t^{(n)}) \) for \( n=1,\ldots,N \) particles, resampling at each step. The likelihood estimate is the product of normalized weights:
   $$
   \hat{p}(S_{1:T} \mid \Theta) = \prod_{t=1}^T \left(\frac{1}{N} \sum_{n=1}^N w_t^{(n)}\right).
   $$

2. **MCMC Step (Metropolis-Hastings):**
   We have a current parameter draw \( \Theta^{(m)} \). Propose \( \Theta^* \) from a proposal distribution \( q(\Theta^* \mid \Theta^{(m)}) \). Accept or reject based on the posterior ratio:
   $$
   A = \frac{p(\Theta^*) \hat{p}(S_{1:T} \mid \Theta^*) q(\Theta^{(m)} \mid \Theta^*)}{p(\Theta^{(m)}) \hat{p}(S_{1:T} \mid \Theta^{(m)}) q(\Theta^* \mid \Theta^{(m)})}.
   $$
   
   If \( U \sim \text{Uniform}(0,1) \), accept if \( U < \min(1, A) \), otherwise reject and set \( \Theta^{(m+1)} = \Theta^{(m)} \).

After burn-in, the chain \( \{\Theta^{(m)}\} \) approximates the posterior distribution over parameters.

---

## Code Structure

The code is organized as follows:

```
bayesian_nonlinear_pricing/
    __init__.py          # Package initialization
    data_generation.py   # Synthetic data generation
    model_params.py      # Parameter management, priors, structure
    hmm.py               # Hidden Markov Model utilities (forward-backward, sampling)
    statespace.py        # Nonlinear state & observation models
    filters.py           # Regime-Switching UKF implementation
    inference.py         # Particle filter for latent states
    utils.py             # Numeric utilities (sigma points, density evaluation)
    pmcmc.py             # Particle MCMC integration for parameter inference
    main.py              # Example script: simulate data, run PMCMC, estimate states
```

### Workflow

1. **Data Generation:**  
   Use `data_generation.py` to produce synthetic \( (S_t, R_t, v_t) \) from known parameters. This serves as ground truth for testing inference methods.
  
2. **Initialization:**  
   Draw initial parameters from specified priors in `model_params.py`.
  
3. **PMCMC Inference:**  
   `pmcmc.py` runs a PMCMC algorithm. Each iteration:
   - Propose new parameters \( \Theta^* \).
   - Run particle filter to compute \( \hat{p}(S_{1:T} \mid \Theta^*) \).
   - Accept/reject the proposal, forming a posterior chain.
  
4. **Filtering with UKF:**  
   Once parameters are estimated, use `filters.py` (RegimeSwitchingUKF) to produce smoothed volatility and regime estimates given the posterior means or samples of \( \Theta \).

---

## Running the Code

**Requirements:** `Python 3`, `numpy`, `scipy`.

```bash
cd bayesian_nonlinear_pricing
python3 main.py
```

The `main.py` script:

1. Generates synthetic data.
2. Samples initial parameters from priors.
3. Runs PMCMC for a specified number of iterations.
4. Retrieves posterior parameter estimates.
5. Performs UKF-based state filtering with the estimated parameters.

---

## Interpreting Results

- **Parameter Estimates:**  
  The PMCMC chain provides draws from the posterior of \( \Theta \). Summaries (mean, median, credible intervals) characterize the uncertainty around model parameters.
  
- **Volatility Filtering:**  
  The final UKF step provides an estimated volatility path \( \hat{v}_t \) that can be compared to the true simulated volatility for validation.
  
- **Regime Probabilities:**  
  At each time \( t \), the filtered regime probabilities \( P(R_t=i \mid S_{1:t}) \) reveal how the algorithm identifies shifts in market states.

---

## Extensions and Advanced Topics

1. **Multi-Factor Volatility Models:**
   Extend \( \mathbf{x}_t \) to include multiple state factors (e.g., jump intensities, risk premia). The UKF and particle filters generalize naturally to higher dimensions.
  
2. **Non-Gaussian Observation Models:**
   If returns exhibit heavy tails or skewness, replace Gaussian assumptions with more flexible distributions (e.g., Student-t), requiring appropriate filtering methods (e.g., particle filters only).
  
3. **Time-Varying Regime Dynamics:**
   Consider more complex regime dynamics or Bayesian updating of the HMM parameters over time.
  
4. **Efficient Proposals and Tuning:**
   The current PMCMC uses simplistic random-walk proposals. Advanced strategies (e.g., adaptive MCMC, Hamiltonian Monte Carlo) could improve mixing and efficiency.

---

## References

- **Particle MCMC:**
  - Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov chain Monte Carlo methods. *J. Royal Statistical Society: Series B*, 72(3), 269–342.
  
- **Heston Volatility Model:**
  - Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6(2), 327–343.
  
- **Unscented Kalman Filter:**
  - Julier, S.J., & Uhlmann, J.K. (1997). A new extension of the Kalman filter to nonlinear systems. *Proceedings of AeroSense: The 11th Int. Symp. on Aerospace/Defense Sensing, Simulation and Controls*.
  
- **Hidden Markov Models in Finance:**
  - Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
