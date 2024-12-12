import numpy as np
from statespace import observation_likelihood
from model_params import half_cauchy_logpdf, dirichlet_logpdf, ModelParameters
from scipy.stats import norm, gamma

class ParticleFilter:
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
            new_R = np.zeros(N, dtype=int)
            for i in range(N):
                new_R[i] = np.random.choice(M, p=self.model_params.Pi[R_part[i],:])

            new_v = np.zeros(N)
            for i in range(N):
                mp = self.model_params.get_params_for_regime(new_R[i])
                v_prev = v_part[i]
                noise = np.random.randn()
                new_v[i] = v_prev + mp['kappa_v']*(mp['theta_v']-v_prev) + mp['sigma_v']*np.sqrt(max(v_prev,1e-12))*noise
                new_v[i] = max(new_v[i],1e-12)

            w = np.zeros(N)
            for i in range(N):
                mp = self.model_params.get_params_for_regime(new_R[i])
                w[i] = observation_likelihood(self.S[t], self.S[t-1], np.array([new_v[i]]), mp)
            w_sum = w.sum()
            if w_sum<1e-50:
                w = np.ones(N)/N
                loglik_contrib = -1e10
            else:
                w /= w_sum
                loglik_contrib = np.log(w_sum/N)
            idx = np.random.choice(N,size=N,p=w)
            v_part = new_v[idx]
            R_part = new_R[idx]
            weights = np.ones(N)/N

            v_store[t] = v_part
            R_store[t] = R_part

        return v_store, R_store

class ParticleMCMC:
    def __init__(self, S, model_params_init, N=200, iterations=1000, burn_in=500, seed=999):
        self.S = S
        self.T = len(S)
        self.model_params = model_params_init
        self.N = N
        self.iterations = iterations
        self.burn_in = burn_in
        np.random.seed(seed)

    def log_prior(self, model_params):
        lp = 0
        # Priors:
        # kappa_v ~ Gamma(2,0.5)
        # theta_v ~ N(0.03,0.01)
        # sigma_v ~ HalfCauchy(0.1)
        # mu ~ N(0,0.001)
        # sigma_level ~ HalfCauchy(0.05)
        for rp in model_params.regime_params:
            lp += gamma(a=2,scale=0.5).logpdf(rp['kappa_v'])
            lp += norm(loc=0.03, scale=0.01).logpdf(rp['theta_v'])
            lp += half_cauchy_logpdf(rp['sigma_v'],0.1)
            lp += norm(loc=0,scale=0.001).logpdf(rp['mu'])
            lp += half_cauchy_logpdf(rp['sigma_level'],0.05)

        # Pi rows ~ Dirichlet(5,...,5)
        alpha = np.ones(model_params.M)*5
        for i in range(model_params.M):
            lp += dirichlet_logpdf(model_params.Pi[i],alpha)

        return lp

    def particle_filter_loglik(self, model_params):
        # Compute PF log-likelihood
        M = model_params.M
        T = self.T
        N = self.N
        np.random.seed(777)
        v_part = np.ones(N)*0.02
        R_part = np.random.choice(M,size=N)
        weights = np.ones(N)/N
        loglik = 0.0

        for t in range(1,T):
            new_R = np.zeros(N, dtype=int)
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
                w[i] = observation_likelihood(self.S[t], self.S[t-1], np.array([new_v[i]]), mp)
            w_sum = w.sum()
            if w_sum<1e-50:
                return -np.inf
            w /= w_sum
            loglik += np.log(w_sum/N)
            idx = np.random.choice(N,size=N,p=w)
            v_part = new_v[idx]
            R_part = new_R[idx]

        return loglik

    def proposal(self, params):
        new_params = params.copy()
        for rp in new_params.regime_params:
            rp['kappa_v'] = abs(rp['kappa_v'] + 0.01*np.random.randn())
            rp['theta_v'] = rp['theta_v'] + 0.001*np.random.randn()
            rp['sigma_v'] = abs(rp['sigma_v'] + 0.01*np.random.randn())
            rp['mu'] = rp['mu'] + 0.0001*np.random.randn()
            rp['sigma_level'] = abs(rp['sigma_level'] + 0.001*np.random.randn())

        Pi_new = new_params.Pi + 0.01*np.random.randn(new_params.M,new_params.M)
        Pi_new = np.abs(Pi_new)
        Pi_new = Pi_new/Pi_new.sum(axis=1, keepdims=True)
        new_params.Pi = Pi_new
        return new_params

    def run_pmcmc(self):
        chain = []
        log_posts = []
        model_current = self.model_params
        log_prior_current = self.log_prior(model_current)
        log_lik_current = self.particle_filter_loglik(model_current)
        if np.isinf(log_lik_current):
            # If initial model gives infinite likelihood, resample
            model_current = ModelParameters.sample_prior(M=model_current.M)
            log_prior_current = self.log_prior(model_current)
            log_lik_current = self.particle_filter_loglik(model_current)
        log_post_current = log_prior_current + log_lik_current

        for it in range(self.iterations):
            model_proposed = self.proposal(model_current)
            log_prior_prop = self.log_prior(model_proposed)
            if np.isinf(log_prior_prop):
                chain.append(model_current)
                log_posts.append(log_post_current)
                continue
            log_lik_prop = self.particle_filter_loglik(model_proposed)
            log_post_prop = log_prior_prop + log_lik_prop
            a = np.exp(log_post_prop - log_post_current)
            if np.random.rand()<a:
                model_current = model_proposed
                log_post_current = log_post_prop
            chain.append(model_current)
            log_posts.append(log_post_current)

        return chain[self.burn_in:], log_posts[self.burn_in:]
