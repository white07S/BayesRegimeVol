import numpy as np
from .statespace import observation_likelihood

class ParticleFilter:
    """
    Particle filter for state inference given fixed parameters.
    This is used inside PMCMC to estimate the likelihood.
    """
    def __init__(self, N, model_params, S, seed=123):
        self.N = N
        self.model_params = model_params
        self.S = S
        self.T = len(S)
        self.state_dim = 1
        np.random.seed(seed)

    def run_filter(self):
        M = self.model_params.M
        T = self.T
        N = self.N
        v_part = np.ones(N)*0.02
        R_part = np.random.choice(M,size=N)
        weights = np.ones(N)/N
        v_store = np.zeros((T,N))
        R_store = np.zeros((T,N), dtype=int)
        v_store[0] = v_part
        R_store[0] = R_part

        for t in range(1,T):
            # Propagate regimes:
            new_R = np.zeros(N, dtype=int)
            for i in range(N):
                new_R[i] = np.random.choice(M, p=self.model_params.Pi[R_part[i], :])

            # Propagate volatility
            new_v = np.zeros(N)
            for i in range(N):
                mp = self.model_params.get_params_for_regime(new_R[i])
                noise = np.random.randn()
                new_v[i] = v_part[i] + mp['kappa_v']*(mp['theta_v']-v_part[i]) + mp['sigma_v']*np.sqrt(max(v_part[i],1e-12))*noise
                new_v[i] = max(new_v[i],1e-12)

            # Compute weights
            w = np.zeros(N)
            for i in range(N):
                mp = self.model_params.get_params_for_regime(new_R[i])
                w[i] = observation_likelihood(self.S[t], self.S[t-1], np.array([new_v[i]]), mp)
            w_sum = w.sum()
            if w_sum<1e-50:
                w = np.ones(N)/N
            else:
                w /= w_sum

            # Resample
            idx = np.random.choice(N,size=N,p=w)
            v_part = new_v[idx]
            R_part = new_R[idx]

            v_store[t] = v_part
            R_store[t] = R_part

        return v_store, R_store

def particle_filter_loglik(N, model_params, S):
    # Compute particle filter log-likelihood
    M = model_params.M
    T = len(S)
    np.random.seed(777)
    v_part = np.ones(N)*0.02
    R_part = np.random.choice(M,size=N)
    loglik = 0.0
    for t in range(1,T):
        new_R = np.zeros(N,dtype=int)
        for i in range(N):
            new_R[i] = np.random.choice(M, p=model_params.Pi[R_part[i],:])
        new_v = np.zeros(N)
        for i in range(N):
            mp = model_params.get_params_for_regime(new_R[i])
            noise = np.random.randn()
            new_v[i] = v_part[i] + mp['kappa_v']*(mp['theta_v']-v_part[i]) + mp['sigma_v']*np.sqrt(max(v_part[i],1e-12))*noise
            new_v[i] = max(new_v[i],1e-12)
        w = np.zeros(N)
        for i in range(N):
            mp = model_params.get_params_for_regime(new_R[i])
            w[i] = observation_likelihood(S[t], S[t-1], np.array([new_v[i]]), mp)
        w_sum = w.sum()
        if w_sum<1e-50:
            return -np.inf
        w /= w_sum
        loglik += np.log(w_sum/N)
        idx = np.random.choice(N,size=N,p=w)
        v_part = new_v[idx]
        R_part = new_R[idx]
    return loglik
