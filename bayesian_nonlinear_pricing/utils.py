import numpy as np
from scipy.linalg import cholesky, cho_solve

def ensure_pos_def(mat):
    epsilon = 1e-12
    for _ in range(10):
        try:
            cholesky(mat)
            return mat
        except:
            mat = mat + np.eye(mat.shape[0])*epsilon
            epsilon *= 10
    return mat

def unscented_transform(mean, cov, alpha=1e-3, beta=2.0, kappa=0):
    n = len(mean)
    lam = alpha**2*(n+kappa)-n
    S = cholesky(ensure_pos_def((n+lam)*cov), lower=True)
    sigmas = np.zeros((2*n+1, n))
    sigmas[0] = mean
    for i in range(n):
        sigmas[i+1] = mean + S[i]
        sigmas[n+i+1] = mean - S[i]

    Wm = np.full(2*n+1, 1/(2*(n+lam)))
    Wc = np.full(2*n+1, 1/(2*(n+lam)))
    Wm[0] = lam/(n+lam)
    Wc[0] = lam/(n+lam)+(1 - alpha**2 + beta)

    return sigmas, Wm, Wc

def multivariate_gauss_density(x, mean, cov):
    k = len(mean)
    diff = x - mean
    chol = cholesky(ensure_pos_def(cov), lower=True)
    sol = cho_solve((chol, True), diff)
    exponent = -0.5 * np.dot(diff, sol)
    det = np.prod(np.diag(chol))
    norm_const = (2*np.pi)**(-k/2) * (1/det)
    return norm_const * np.exp(exponent)

def half_cauchy_logpdf(x, scale):
    if x<=0:
        return -np.inf
    return np.log(2/(np.pi)) + np.log(scale) - np.log(x**2+scale**2)

def dirichlet_logpdf(x, alpha):
    if np.any(x<=0):
        return -np.inf
    from math import lgamma
    sum_alpha = np.sum(alpha)
    log_f = np.sum((alpha-1)*np.log(x))
    log_B = np.sum([lgamma(a) for a in alpha]) - lgamma(sum_alpha)
    return log_f - log_B
