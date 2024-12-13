import numpy as np
from scipy.stats import gamma, norm
from .utils import half_cauchy_logpdf, dirichlet_logpdf

class ModelParameters:
    """
    Holds model parameters for each regime and the transition matrix Pi.
    regime_params is a list of dicts: [{'mu':..., 'sigma_level':..., 'kappa_v':..., 'theta_v':..., 'sigma_v':...}, ...]
    Pi is MxM transition matrix.
    """
    def __init__(self, M=2, regime_params=None, Pi=None):
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
        # Priors chosen as described:
        # kappa_v ~ Gamma(2,0.5)
        # theta_v ~ N(0.03,0.01)
        # sigma_v ~ HalfCauchy(scale=0.1)
        # mu ~ N(0,0.001)
        # sigma_level ~ HalfCauchy(scale=0.05)
        # Pi rows ~ Dirichlet([5,...,5])
        res = []
        def half_cauchy_draw(scale):
            u = np.random.rand()
            return abs(scale*np.tan(np.pi*u/2))
        for _ in range(M):
            kappa_v = gamma(a=2, scale=0.5).rvs()
            theta_v = norm(loc=0.03, scale=0.01).rvs()
            sigma_v = half_cauchy_draw(0.1)
            mu = norm(loc=0, scale=0.001).rvs()
            sigma_level = half_cauchy_draw(0.05)
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
        reg_copy = [rp.copy() for rp in self.regime_params]
        return ModelParameters(M=self.M, regime_params=reg_copy, Pi=self.Pi.copy())

    def log_prior(self):
        # Using the stated priors:
        lp = 0
        from math import inf
        from .utils import half_cauchy_logpdf, dirichlet_logpdf
        # Parameter priors:
        # kappa_v: Gamma(2,0.5)
        # theta_v: N(0.03,0.01)
        # sigma_v: HalfCauchy(0.1)
        # mu: N(0,0.001)
        # sigma_level: HalfCauchy(0.05)
        # Pi: Dirichlet([5,5,...])
        from scipy.stats import gamma, norm
        alpha = np.ones(self.M)*5
        for rp in self.regime_params:
            lp += gamma(a=2,scale=0.5).logpdf(rp['kappa_v'])
            lp += norm(loc=0.03,scale=0.01).logpdf(rp['theta_v'])
            lp += half_cauchy_logpdf(rp['sigma_v'],0.1)
            lp += norm(loc=0,scale=0.001).logpdf(rp['mu'])
            lp += half_cauchy_logpdf(rp['sigma_level'],0.05)
        for i in range(self.M):
            lp += dirichlet_logpdf(self.Pi[i],alpha)
        if np.isinf(lp):
            return -inf
        return lp
