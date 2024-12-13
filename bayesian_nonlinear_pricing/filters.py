import numpy as np
from .utils import unscented_transform, ensure_pos_def
from .statespace import state_transition

class RegimeSwitchingUKF:
    """
    Regime-switching UKF in an IMM framework.
    State: v_t (one-dimensional)
    Observation: y_t = log(S_t)
    The measurement model is complicated because the measurement variance depends on v.
    We approximate measurement noise covariance R using the predicted mean v.
    This leads to zero direct state update from measurement since h(x) is constant wrt x.
    The update in state estimate happens indirectly via regime probability updates.
    """
    def __init__(self, model_params, state_dim=1, alpha=1e-3, beta=2.0, kappa=0):
        self.model_params = model_params
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.M = self.model_params.M

    def predict_update(self, S_t, S_prev, mean_prev, cov_prev, regime_probs):
        # Predict step for each regime
        means_pred = []
        covs_pred = []
        for r in range(self.M):
            mp = self.model_params.get_params_for_regime(r)
            m_pred, P_pred = self.ukf_time_update(mean_prev[r], cov_prev[r], mp)
            means_pred.append(m_pred)
            covs_pred.append(P_pred)

        # Mix over regimes for prediction step:
        # Predicted regime probabilities
        pred_regime_probs = regime_probs @ self.model_params.Pi

        # Compute observation likelihood for each regime using predicted mean:
        # h(x)=log(S_prev)+mu, but var = v_mean * sigma_level^2
        # We'll set R = sigma_level^2 * mean_v_predicted for that regime
        like_per_regime = []
        for i in range(self.M):
            mp = self.model_params.get_params_for_regime(i)
            # Compute predicted measurement distribution
            # Obtain mean_v from means_pred[i]
            v_mean = means_pred[i][0]
            mu = mp['mu']
            sigma_level = mp['sigma_level']
            # The observation is: y=log(S_t)
            # Predicted mean: y_mean=log(S_prev)+mu
            # var_y=v_mean*sigma_level^2
            y_obs = np.log(S_t)
            y_mean_pred = np.log(S_prev)+mu
            var_y = max(v_mean*sigma_level**2,1e-15)
            diff = y_obs - y_mean_pred
            # Gaussian pdf:
            pdf_val = (1.0/np.sqrt(2*np.pi*var_y))*np.exp(-0.5*(diff**2)/var_y)
            like_per_regime.append(pdf_val)

        like_per_regime = np.array(like_per_regime)
        numer = like_per_regime * pred_regime_probs
        denom = numer.sum()
        if denom<1e-50:
            denom = 1e-50
        posterior_regime_probs = numer/denom

        # Measurement update for states:
        # Since h(x) is constant wrt x, Pxy=0 and no direct update.
        # means_post and covs_post are just means_pred and covs_pred in this simplified approach.
        means_post = means_pred
        covs_post = covs_pred

        return means_post, covs_post, posterior_regime_probs, denom

    def ukf_time_update(self, mean, cov, regime_params):
        sigmas, Wm, Wc = unscented_transform(mean, cov, self.alpha, self.beta, self.kappa)
        x_pred_list = []
        for s in sigmas:
            x_pred_list.append(state_transition(s, regime_params))
        x_pred = np.array(x_pred_list)
        mean_pred = (Wm[:,None]*x_pred).sum(axis=0)
        diff = x_pred - mean_pred
        P_pred = np.zeros_like(cov)
        for i in range(len(Wc)):
            P_pred += Wc[i]*np.outer(diff[i], diff[i])
        P_pred = ensure_pos_def(P_pred)
        return mean_pred, P_pred
