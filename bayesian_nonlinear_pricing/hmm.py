import numpy as np

def sample_regime_path(Pi, T, init_probs=None, random_state=None):
    if random_state is None:
        random_state = np.random
    M = Pi.shape[0]
    if init_probs is None:
        init_probs = np.ones(M)/M
    R = np.zeros(T, dtype=int)
    R[0] = random_state.choice(M, p=init_probs)
    for t in range(1,T):
        R[t] = random_state.choice(M, p=Pi[R[t-1],:])
    return R

def forward_backward_hmm(Pi, regime_likelihoods, init_probs=None):
    # regime_likelihoods[t,i] = p(y_t|R_t=i)
    T, M = regime_likelihoods.shape
    if init_probs is None:
        init_probs = np.ones(M)/M
    alpha = np.zeros((T,M))
    c = np.zeros(T)
    # Forward
    alpha[0] = init_probs*regime_likelihoods[0]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1,T):
        alpha[t] = (alpha[t-1].dot(Pi))*regime_likelihoods[t]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]

    # Backward
    beta = np.zeros((T,M))
    beta[-1] = 1.0
    for t in reversed(range(T-1)):
        beta[t] = Pi.dot((beta[t+1]*regime_likelihoods[t+1]))/c[t+1]

    gamma = alpha*beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, alpha, beta, c
