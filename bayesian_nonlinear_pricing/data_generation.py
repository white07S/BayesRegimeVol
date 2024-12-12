import numpy as np
from model_params import ModelParameters
from hmm import sample_regime_path
from statespace import volatility_dynamics, price_dynamics

def generate_synthetic_data(T=1000, M=2, seed=42):
    np.random.seed(seed)
    regime_params = [
        {'mu':0.001,'sigma_level':0.02,'kappa_v':0.5,'theta_v':0.02,'sigma_v':0.1},
        {'mu':0.0,'sigma_level':0.05,'kappa_v':1.0,'theta_v':0.05,'sigma_v':0.2}
    ]
    Pi = np.array([[0.95,0.05],
                   [0.10,0.90]])
    model = ModelParameters(M=M, regime_params=regime_params, Pi=Pi)

    R = sample_regime_path(model.Pi, T)
    S = np.zeros(T)
    v = np.zeros(T)
    S[0] = 100.0
    v[0] = 0.02

    for t in range(1,T):
        rp = model.get_params_for_regime(R[t])
        v[t] = volatility_dynamics(v[t-1], rp['kappa_v'], rp['theta_v'], rp['sigma_v'])
        S[t] = price_dynamics(S[t-1], v[t], rp['mu'], rp['sigma_level'])

    return S, R, v, model
