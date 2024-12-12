import numpy as np
from inference import ParticleFilter
from model_params import half_cauchy_logpdf, dirichlet_logpdf, ModelParameters
from statespace import observation_likelihood
from scipy.stats import norm, gamma

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
        for rp in model_params.regime_params:
            lp += gamma(a=2,scale=0.5).logpdf(rp['kappa_v'])
            lp += norm(loc=0.03, scale=0.01).logpdf(rp['theta_v'])
            # half_cauchy
            def hc_lp(x,scale):
                if x<=0:
                    return -np.inf
                return np.log(2/(np.pi)) + np.log(scale) - np.log(x**2+scale**2)
            lp += hc_lp(rp['sigma_v'],0.1)
            lp += norm(loc=0,scale=0.001).logpdf(rp['mu'])
            lp += hc_lp(rp['sigma_level'],0.05)
        alpha = np.ones(model_params.M)*5
        def dir_lp(x,a):
            if np.any(x<=0):
                return -np.inf
            from math import lgamma
            sum_a = np.sum(a)
            res = np.sum((a-1)*np.log(x))
            log_B = np.sum([lgamma(ai) for ai in a]) - lgamma(sum_a)
            return res - log_B
        for i in range(model_params.M):
            lp += dir_lp(model_params.Pi[i], alpha)
        return lp

    def particle_filter_loglik(self, model_params):
        M = model_params.M
        T = self.T
        N = self.N
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
        model_current = self.model_params
        log_prior_current = self.log_prior(model_current)
        log_lik_current = self.particle_filter_loglik(model_current)
        if np.isinf(log_lik_current):
            model_current = ModelParameters.sample_prior(M=model_current.M)
            log_prior_current = self.log_prior(model_current)
            log_lik_current = self.particle_filter_loglik(model_current)
        log_post_current = log_prior_current+log_lik_current

        chain = []
        log_posts = []
        for it in range(self.iterations):
            model_proposed = self.proposal(model_current)
            log_prior_prop = self.log_prior(model_proposed)
            if np.isinf(log_prior_prop):
                chain.append(model_current)
                log_posts.append(log_post_current)
                continue
            log_lik_prop = self.particle_filter_loglik(model_proposed)
            log_post_prop = log_prior_prop+log_lik_prop
            a = np.exp(log_post_prop - log_post_current)
            if np.random.rand()<a:
                model_current = model_proposed
                log_post_current = log_post_prop
            chain.append(model_current)
            log_posts.append(log_post_current)
        return chain[self.burn_in:], log_posts[self.burn_in:]
