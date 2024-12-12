import numpy as np
from data_generation import generate_synthetic_data
from model_params import ModelParameters
from pmcmc import ParticleMCMC
from filters import RegimeSwitchingUKF

def main():
    # Generate Synthetic Data
    T = 300
    S, R_true, v_true, true_model = generate_synthetic_data(T=T, M=2, seed=42)

    # Sample initial parameters from prior
    init_model = ModelParameters.sample_prior(M=2)

    # Run PMCMC
    pmcmc = ParticleMCMC(S, init_model, N=200, iterations=500, burn_in=200, seed=123)
    chain, log_posts = pmcmc.run_pmcmc()

    # Take last sample as final parameters estimate
    final_params = chain[-1]

    # Run Regime-Switching UKF for state estimation
    state_dim = 1
    M = final_params.M
    mean_prev = [np.array([0.02]) for _ in range(M)]
    cov_prev = [np.array([[0.001]]) for _ in range(M)]
    regime_probs = np.ones(M)/M
    ukf = RegimeSwitchingUKF(final_params, state_dim=state_dim)

    v_est = np.zeros(T)
    v_est[0] = 0.02

    # Filtering states:
    for t in range(1,T):
        means_post, covs_post, regime_probs, _ = ukf.predict_update(S[t], S[t-1], mean_prev, cov_prev, regime_probs)
        mean_prev = means_post
        cov_prev = covs_post
        v_t_est = 0
        for i in range(M):
            v_t_est += regime_probs[i]*means_post[i][0]
        v_est[t] = v_t_est

    print("True final volatility:", v_true[-1])
    print("Estimated final volatility:", v_est[-1])
    print("Final regime probabilities:", regime_probs)
    print("Done.")

if __name__=="__main__":
    main()
