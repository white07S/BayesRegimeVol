import numpy as np
from scipy.stats import gamma, norm

# We will manually compute half-Cauchy and Dirichlet priors to avoid placeholders.

def half_cauchy_logpdf(x, scale):
    # half-Cauchy pdf: f(x)=2/(pi)*scale/(x^2+scale^2), x>0
    if x<=0:
        return -np.inf
    return np.log(2/(np.pi)) + np.log(scale) - np.log(x**2+scale**2)

def dirichlet_logpdf(x, alpha):
    # Dirichlet: f(x|alpha)= 1/B(alpha)*∏ x_i^(alpha_i-1)
    # log f = -log B(alpha) + sum((alpha_i-1)*log x_i)
    # B(alpha)=∏ Gamma(alpha_i)/Gamma(sum(alpha_i))
    if np.any(x<=0):
        return -np.inf
    sum_alpha = np.sum(alpha)
    log_f = 0
    log_f += np.sum((alpha-1)*np.log(x))
    # Compute log B(alpha)
    from math import lgamma
    log_B = 0
    log_B = np.sum([lgamma(a) for a in alpha]) - lgamma(sum_alpha)
    log_f -= log_B
    return log_f

class ModelParameters:
    def __init__(self, 
                 M=2,
                 regime_params=None,
                 Pi=None):
        self.M = M
        if regime_params is None:
            regime_params = [
                {'mu':0.001,'sigma_level':0.02,'kappa_v':0.5,'theta_v':0.02,'sigma_v':0.1},
                {'mu':0.0,'sigma_level':0.05,'kappa_v':1.0,'theta_v':0.05,'sigma_v':0.2}
            ]
        self.regime_params = regime_params
        if Pi is None:
            Pi = np.array([[0.95,0.05],
                           [0.10,0.90]])
        self.Pi = Pi

    def get_params_for_regime(self, r):
        return self.regime_params[r]

    @staticmethod
    def sample_prior(M=2):
        # Priors:
        # kappa_v ~ Gamma(a=2,b=0.5)
        # theta_v ~ N(0.03,0.01)
        # sigma_v ~ HalfCauchy(scale=0.1)
        # mu ~ N(0,0.001)
        # sigma_level ~ HalfCauchy(scale=0.05)
        # Pi rows ~ Dirichlet([5,...,5])
        res = []
        for _ in range(M):
            kappa_v = gamma(a=2, scale=0.5).rvs()
            theta_v = norm(loc=0.03, scale=0.01).rvs()
            # half-Cauchy: use inverse transform or just a wide scale normal + abs trick:
            # to ensure it's from half-Cauchy, we can do a trick: Let u=Uniform(0,1), x=scale*tan(pi*u/2)
            def half_cauchy_draw(scale):
                u = np.random.rand()
                return scale*np.tan(np.pi*u/2)
            sigma_v = abs(half_cauchy_draw(0.1))
            mu = norm(loc=0, scale=0.001).rvs()
            sigma_level = abs(half_cauchy_draw(0.05))
            res.append({'mu':mu,'sigma_level':sigma_level,'kappa_v':kappa_v,'theta_v':theta_v,'sigma_v':sigma_v})
        alpha = np.ones(M)*5
        row_list = []
        for i in range(M):
            z = np.random.gamma(5,1,size=M)
            z = z/z.sum()
            row_list.append(z)
        Pi = np.vstack(row_list)
        return ModelParameters(M=M, regime_params=res, Pi=Pi)

    def copy(self):
        reg_copy = []
        for rp in self.regime_params:
            reg_copy.append(rp.copy())
        return ModelParameters(M=self.M, regime_params=reg_copy, Pi=self.Pi.copy())
