import numpy as np

def volatility_dynamics(v_prev, kappa_v, theta_v, sigma_v):
    noise = np.random.randn()
    v_t = v_prev + kappa_v*(theta_v - v_prev) + sigma_v*np.sqrt(max(v_prev,1e-10))*noise
    return max(v_t,1e-12)

def price_dynamics(S_prev, v_t, mu, sigma_level):
    noise = np.random.randn()
    dS = mu + np.sqrt(v_t)*sigma_level*noise
    return S_prev*np.exp(dS)

def state_transition(x_prev, regime_params):
    # x_prev = [v_{t-1}]
    v_prev = x_prev[0]
    v_t = volatility_dynamics(v_prev, regime_params['kappa_v'], regime_params['theta_v'], regime_params['sigma_v'])
    return np.array([v_t])

def observation_likelihood(S_t, S_prev, x_t, regime_params):
    # log(S_t) ~ Normal(log(S_{t-1}) + mu, v_t*sigma_level^2)
    v_t = x_t[0]
    mu = regime_params['mu']
    sigma_level = regime_params['sigma_level']
    mean = np.log(S_prev) + mu
    var = max(v_t*sigma_level**2, 1e-15)
    diff = np.log(S_t) - mean
    # pdf of log(S_t)
    # p(S_t) = (1/(S_t*sqrt(2*pi*var)))*exp(-0.5*(diff^2/var))
    if S_t<=0:
        return 1e-50
    return (1.0/(S_t*np.sqrt(2*np.pi*var)))*np.exp(-0.5*(diff**2)/var)
