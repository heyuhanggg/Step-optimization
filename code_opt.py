import json
import math
from pathlib import Path

import torch
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

# DPM-Solver-compatible timestep optimization with a coupled DU+PS pipeline:
# the same normalized drift-aware profile drives DU initialization and PS refinement.



class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \\hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \\sqrt{\\hat{alpha_n}} * x_0, (1 - \\hat{alpha_n}) * I ).
                Therefore, the notation \\hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \\sqrt{\\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
        
        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \\hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t
        
    def edm_sigma(self, t):
        return self.marginal_std(t)/self.marginal_alpha(t)
    
    def edm_inverse_sigma(self, edmsigma):
        alpha = 1 / torch.sqrt(edmsigma ** 2 + 1)
        sigma = alpha*edmsigma
        lambda_t = torch.log(alpha / sigma)
        t = self.inverse_lambda(lambda_t)
        return t

def interpolate_fn(x, xp, yp):
    """
    Piecewise linear interpolation function y = f(x), implemented in a differentiable way.
    Uses xp and yp as keypoints. For x beyond bounds, uses outermost points.

    Args:
        x: PyTorch tensor with shape [N, C] (batch size N, channels C)
        xp: PyTorch tensor with shape [C, K] (K keypoints)
        yp: PyTorch tensor with shape [C, K]
    
    Returns:
        Function values f(x) with shape [N, C]
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def _extract_tensor_batch(batch):
    """Return the image tensor from common dataloader batch formats."""
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if isinstance(batch, dict):
        for key in ("image", "images", "x", "x0", "data"):
            if key in batch:
                return batch[key]
    raise TypeError("Unsupported calibration batch format.")


def make_uniform_lambda_calibration_times(
    noise_schedule,
    *,
    n_t=64,
    eps=1e-3,
    T=None,
    dtype=torch.float32,
):
    """
    Construct the Algorithm-1 calibration times uniformly in the lambda-domain.

    The manuscript selects {t_i}_{i=1}^{N_t} uniformly in
        lambda(t) = log(alpha_t / sigma_t).
    This helper returns the corresponding time-domain values.
    """
    if int(n_t) < 2:
        raise ValueError("n_t must be at least 2.")
    if T is None:
        T = float(noise_schedule.T)

    t_T = torch.tensor([float(T)], dtype=torch.float64)
    t_eps = torch.tensor([float(eps)], dtype=torch.float64)
    lambda_T = noise_schedule.marginal_lambda(t_T).item()
    lambda_eps = noise_schedule.marginal_lambda(t_eps).item()
    lambda_values = torch.linspace(
        lambda_T, lambda_eps, int(n_t), dtype=torch.float64
    )
    return noise_schedule.inverse_lambda(lambda_values).to(dtype=dtype)


def estimate_ltheta_finite_difference(
    denoiser,
    calibration_data,
    noise_schedule,
    t_values,
    *,
    k_directions=16,
    eps_scale=1e-3,
    quantile=0.95,
    batch_size=64,
    max_batches=None,
    device="cuda",
):
    """
    Estimate the empirical local-sensitivity profile used by manuscript Algorithm 1.

    Paper-matched details:
      1) one fixed calibration batch X_0 of size B is drawn once and reused for
         every calibration time t_i;
      2) each sample has its own perturbation scale
             eps_i^(b) = eps_scale * RMS(x_{t_i}^(b));
      3) K unit-l2 directions are drawn independently for every sample/time;
      4) Lhat_theta(t_i) is the q-quantile over all B x K finite-difference
         estimates, with no extra 1/sqrt(d) normalization.

    `t_values` should be uniformly spaced in the lambda-domain. Use
    `make_uniform_lambda_calibration_times(...)` to construct them exactly as in
    Algorithm 1.
    """
    t_tensor = torch.as_tensor(t_values, dtype=torch.float32).flatten()
    if t_tensor.numel() < 1:
        raise ValueError("t_values must contain at least one calibration time.")

    denoiser_device = torch.device(device)

    if torch.is_tensor(calibration_data):
        x0_pool = calibration_data.detach().to(denoiser_device)
    else:
        batches = []
        for i, batch in enumerate(calibration_data):
            if max_batches is not None and i >= max_batches:
                break
            batches.append(_extract_tensor_batch(batch).detach().cpu())
        if not batches:
            raise ValueError("calibration_data produced no batches.")
        x0_pool = torch.cat(batches, dim=0).to(denoiser_device)

    B = int(batch_size)
    if B < 1:
        raise ValueError("batch_size must be positive.")
    if x0_pool.shape[0] < B:
        raise ValueError(
            f"Algorithm 1 requires B={B} calibration samples, but only "
            f"{x0_pool.shape[0]} are available."
        )

    # Algorithm 1: draw ONE fixed calibration batch X_0 and reuse it for all t_i.
    fixed_idx = torch.randperm(x0_pool.shape[0], device=denoiser_device)[:B]
    x0 = x0_pool.index_select(0, fixed_idx)
    view_shape = (B,) + (1,) * (x0.ndim - 1)
    reduce_dims = tuple(range(1, x0.ndim))

    l_values = []
    with torch.no_grad():
        for t in t_tensor.to(denoiser_device):
            # One independent forward-noise realization per calibration sample.
            z = torch.randn_like(x0)
            t_batch = torch.full(
                (B,), float(t.item()), device=denoiser_device, dtype=torch.float32
            )
            alpha = noise_schedule.marginal_alpha(t_batch).to(denoiser_device).view(view_shape)
            sigma = noise_schedule.marginal_std(t_batch).to(denoiser_device).view(view_shape)
            x_t = alpha * x0 + sigma * z

            # Paper: eps_i^(b) = eps_scale * RMS(x_{t_i}^(b)), per sample.
            eps_b = float(eps_scale) * torch.sqrt(
                torch.mean(x_t.float() ** 2, dim=reduce_dims)
            ).clamp_min(1e-12)
            eps_view = eps_b.view(view_shape)

            directional = []
            for _ in range(int(k_directions)):
                u = torch.randn_like(x_t)
                u = u / u.reshape(B, -1).norm(dim=1).clamp_min(1e-12).view(view_shape)

                y_plus = denoiser(x_t + eps_view * u, t_batch)
                y_minus = denoiser(x_t - eps_view * u, t_batch)

                # One finite-difference sensitivity estimate per sample and direction.
                diff_b = (y_plus - y_minus).reshape(B, -1).norm(dim=1) / (2.0 * eps_b)
                directional.append(diff_b.detach().float().cpu())

            # Quantile over all B x K sample-direction estimates.
            all_vals = torch.cat(directional, dim=0)
            l_values.append(torch.quantile(all_vals, float(quantile)))

    return {
        "t": t_tensor.cpu(),
        "L": torch.stack(l_values).cpu(),
        "meta": {
            "k_directions": int(k_directions),
            "eps_scale": float(eps_scale),
            "quantile": float(quantile),
            "batch_size": B,
            "fixed_calibration_batch_across_t": True,
            "per_sample_rms_eps": True,
            "aggregation": "quantile_over_BxK",
            "normalized_by_sqrt_dim": False,
        },
    }


def save_ltheta_profile(profile, path):
    """Save an Algorithm-1 sensitivity profile for StepOptim.set_Ltheta_from_file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, output_path)


class StepOptim(object):
    """
    Timestep optimization for diffusion sampling using local order of approximation.
    Supports various initialization strategies and optional data-driven Lipschitz bounds.
    """
    
    def __init__(
            self,
            ns,
            max_t,
            *,
            l_theta=None,
            surrogate_alpha=0.3,
            paper_cpath_dense_M=4096,
            # Legacy path-blend arguments are retained only for backward compatibility.
            # They are not used by the paper Eq. (22) objective below.

            optimizer_verbose=0,
        ):
        """
        Args:
            ns: NoiseScheduleVP instance defining the noise schedule
            max_t: Maximum time t_T (1.0 for VP models, 80.0 for EDM models)
        """
        super().__init__()
        self.ns = ns
        self.T = max_t
        self.Ltheta_fn = None
        # Paper Eq. (22): additive normalized empirical surrogate.
        self.L_theta = None if l_theta is None else float(l_theta)
        self.Ltheta_mode = "unset" if self.L_theta is None else "const"
        self.surrogate_alpha = float(surrogate_alpha)
        self.paper_cpath_dense_M = int(paper_cpath_dense_M)

        # Legacy path-blend parameters are kept only so existing caller code does not break.
        # They are intentionally unused by sel_lambdas_lof_obj() in this paper-matched file.
        self.optimizer_verbose = int(optimizer_verbose)
    def set_Ltheta_from_file(self, path, smooth_ema_tau=0.0, normalize_geom=False, clip=(None, None)):
        """
        Load empirical local sensitivity data and construct an L_theta(t) interpolator.
        
        Args:
            path: Path to torch.save file containing {'t': Tensor[M], 'L': Tensor[M], 'meta': dict}
            smooth_ema_tau: EMA smoothing strength in [0, 1] (0=no smoothing)
            normalize_geom: Normalize curve geometric mean to 1 (removes global scale, keeps shape)
            clip: (lo, hi) clipping range, e.g. (None, 1e3)
        """
        payload = torch.load(path, map_location="cpu")
        t_nodes = payload["t"].float().numpy()   # Ascending t values
        L_nodes = payload["L"].float().numpy()

        # Sort and deduplicate
        idx = np.argsort(t_nodes)
        t_nodes = t_nodes[idx]; L_nodes = L_nodes[idx]
        mask = np.concatenate([[True], np.diff(t_nodes) > 1e-12])
        t_nodes = t_nodes[mask]; L_nodes = L_nodes[mask]

        # Optional: EMA smoothing (more stable in log domain)
        if smooth_ema_tau and smooth_ema_tau > 0:
            logL = np.log(np.clip(L_nodes, 1e-12, None))
            out = np.empty_like(logL); m = logL[0]
            for i in range(len(logL)):
                m = smooth_ema_tau * logL[i] + (1 - smooth_ema_tau) * m
                out[i] = m
            L_nodes = np.exp(out)

        # Optional: Geometric mean normalization (removes global scale)
        if normalize_geom:
            gmean = np.exp(np.mean(np.log(np.clip(L_nodes, 1e-12, None))))
            L_nodes = L_nodes / max(gmean, 1e-12)

        # Optional: Clipping
        lo, hi = clip
        if lo is not None: L_nodes = np.maximum(L_nodes, lo)
        if hi is not None: L_nodes = np.minimum(L_nodes, hi)

        # Linear interpolation (use boundary values for out-of-bounds)
        def _interp_L_of_t(t_query):
            tq = np.asarray(t_query, dtype=np.float64)
            return np.interp(tq, t_nodes, L_nodes, left=L_nodes[0], right=L_nodes[-1])

        self.Ltheta_mode = "data"
        self.Ltheta_fn   = _interp_L_of_t

    def alpha(self, t):
        """Compute alpha_t from time t."""
        t = torch.as_tensor(t, dtype=torch.float64)
        return self.ns.marginal_alpha(t).numpy()
    
    def sigma(self, t):
        """Compute sigma_t from time t."""
        return np.sqrt(1 - self.alpha(t) * self.alpha(t))
    
    def lambda_func(self, t):
        """Compute log-SNR lambda_t = log(alpha_t / sigma_t)."""
        return np.log(self.alpha(t) / self.sigma(t))
    
    # Helper functions for Taylor expansion coefficients
    def H0(self, h):
        return np.exp(h) - 1
    
    def H1(self, h):
        return np.exp(h) * h - self.H0(h)
    
    def H2(self, h):
        return np.exp(h) * h * h - 2 * self.H1(h)
    
    def H3(self, h):
        return np.exp(h) * h * h * h - 3 * self.H2(h)
    
    def inverse_lambda(self, lamb):
        """Compute time t from log-SNR lambda."""
        lamb = torch.as_tensor(lamb, dtype=torch.float64)
        return self.ns.inverse_lambda(lamb)
    
    def edm_sigma(self, t):
        """Compute EDM-style sigma from time t."""
        return np.sqrt(1. / (self.alpha(t) * self.alpha(t)) - 1)
    
    def edm_inverse_sigma(self, edm_sigma):
        """Compute time t from EDM-style sigma."""
        alpha = 1 / np.sqrt(edm_sigma * edm_sigma + 1)
        sigma = alpha * edm_sigma
        lambda_t = np.log(alpha / sigma)
        t = self.inverse_lambda(lambda_t)
        return t

    def _dlog_alpha_dt(self, t_arr, h=1e-4):
        """Numerical derivative of log(alpha_t) with respect to t."""
        t_arr = np.asarray(t_arr, dtype=np.float64)
        t_lo = np.maximum(t_arr - h, 1e-6)
        t_hi = np.minimum(t_arr + h, float(self.T))
        with torch.no_grad():
            l_hi = self.ns.marginal_log_mean_coeff(torch.as_tensor(t_hi, dtype=torch.float64)).numpy()
            l_lo = self.ns.marginal_log_mean_coeff(torch.as_tensor(t_lo, dtype=torch.float64)).numpy()
        return (l_hi - l_lo) / (t_hi - t_lo + 1e-20)

    def _ab_gamma(self, t_arr):
        """
        Compute drift coefficients a(t), b(t) and error growth rate gamma(t).
        gamma(t) = |a(t)| + |b(t)| * L_theta(t)
        """
        t_arr = np.asarray(t_arr, dtype=np.float64)
        with torch.no_grad():
            l = self.ns.marginal_log_mean_coeff(torch.as_tensor(t_arr, dtype=torch.float64)).numpy()
        dl = self._dlog_alpha_dt(t_arr)
        exp2l = np.exp(2.0 * l)
        denom = np.maximum(1.0 - exp2l, 1e-6)
        a = -(exp2l / denom) * dl         # a(t)
        b = (np.exp(l) * dl) / denom      # b(t)

        # Use data-driven L_theta(t) if set, otherwise fall back to an explicit constant.
        if self.Ltheta_mode == "data" and self.Ltheta_fn is not None:
            L_vals = np.asarray(self.Ltheta_fn(t_arr), dtype=np.float64)
        elif self.Ltheta_mode == "const" and self.L_theta is not None:
            L_vals = np.full_like(t_arr, self.L_theta, dtype=np.float64)
        else:
            raise RuntimeError(
                "Empirical L_theta(t) profile has not been loaded. "
                "Call set_Ltheta_from_file(...) before schedule optimization, "
                "or pass an explicit l_theta constant to StepOptim(...)."
            )
        gamma = np.abs(a) + np.abs(b) * L_vals
        return a, b, gamma
    

    def _paper_log_cpath(self, lam_query):
        """
        Numerically evaluate the paper path-amplification profile

            log C_path(t) = integral_t^T gamma(s) ds,
            gamma(t) = |a(t)| + |b(t)| L_theta(t),

        on a dense lambda grid and interpolate it at lam_query.

        This routine intentionally does NOT apply path temperature, linear mixing,
        or log-factor clipping, because those operations are not part of the
        manuscript Eq. (22).  The subsequent Eq. (22) normalization is carried out
        in log space to avoid overflow without changing the mathematical value.
        """
        lam_query = np.asarray(lam_query, dtype=np.float64)
        lambda_T = self.lambda_func(self.T).item()

        # All candidate segment endpoints lie below T and above eps; use their
        # largest lambda as the lower-time end of the dense integration grid.
        lambda_max = float(np.max(lam_query))
        dense_M = max(int(self.paper_cpath_dense_M), 64)
        lam_dense = np.linspace(lambda_T, lambda_max, dense_M, dtype=np.float64)

        # lambda increases as reverse time decreases: t_dense[0] = T and
        # t_dense[-1] is the smallest queried time.
        t_dense = np.asarray(
            self.inverse_lambda(torch.as_tensor(lam_dense, dtype=torch.float64)),
            dtype=np.float64,
        )
        _, _, gamma_dense = self._ab_gamma(t_dense)

        # Positive trapezoidal mass between successive reverse-time nodes.
        # cum_from_T[j] = integral_{t_j}^{T} gamma(s) ds.
        dt = np.abs(np.diff(t_dense))
        seg = 0.5 * (gamma_dense[:-1] + gamma_dense[1:]) * dt
        cum_from_T = np.concatenate([[0.0], np.cumsum(seg)])

        return np.interp(lam_query, lam_dense, cum_from_T)

    def _paper_combined_profile_on_lambda(self, lam_query, data_err_vec=None):
        """Evaluate the SAME normalized DU+PS difficulty profile used by Eq. (22).

        For any lambda-grid ``lam_query``, compute

            zeta_bar(lambda)
              = surrogate_alpha * Norm[Lhat(t) * C_path(t)]
                + Norm[eps_tilde(t)],

        where

            C_path(t) = exp(integral_t^T gamma(s) ds),
            gamma(t)  = |a(t)| + |b(t)| Lhat(t),
            eps_tilde = sigma(t)^2 / alpha(t)   (epsilon_t == 1).

        This helper is deliberately shared by BOTH stages of the paper method:
          1) DU initialization uses g(lambda)=zeta_bar(lambda)*exp(lambda) and
             equidistributes its cumulative mass;
          2) PS refinement uses the same zeta_bar at the current solver nodes in
             the constrained Eq. (20)/(22) objective.

        Therefore DU and PS are coupled through one common empirical temporal
        difficulty profile rather than using unrelated profiles.
        """
        lam = np.asarray(lam_query, dtype=np.float64).reshape(-1)
        if lam.size == 0:
            return np.asarray([], dtype=np.float64)

        t = np.asarray(
            self.inverse_lambda(torch.as_tensor(lam, dtype=torch.float64)),
            dtype=np.float64,
        )

        # Empirical local sensitivity Lhat_theta(t).
        if self.Ltheta_mode == "data" and self.Ltheta_fn is not None:
            L = np.asarray(self.Ltheta_fn(t), dtype=np.float64)
        elif self.Ltheta_mode == "const" and self.L_theta is not None:
            L = np.full_like(t, self.L_theta, dtype=np.float64)
        else:
            raise RuntimeError(
                "Empirical L_theta(t) profile has not been loaded. "
                "Call set_Ltheta_from_file(...) before schedule optimization, "
                "or pass an explicit l_theta constant to StepOptim(...)."
            )
        L = np.maximum(L, 1e-300)

        # Path-sensitive factor C_path(t)=exp(integral_t^T gamma).
        logC = self._paper_log_cpath(lam)
        log_drift = np.log(L) + logC

        # Arithmetic-mean normalization of Lhat*C_path, evaluated stably in log-space.
        shift = float(np.max(log_drift))
        log_mean_drift = shift + math.log(
            max(float(np.mean(np.exp(log_drift - shift))), 1e-300)
        )
        drift_norm = np.exp(log_drift - log_mean_drift)

        # Training-error profile eps_tilde = sigma^2 / alpha (epsilon_t == 1).
        if data_err_vec is None:
            exp_pos = np.exp(np.clip(2.0 * lam, -700.0, 700.0))
            exp_neg = np.exp(np.clip(-2.0 * lam, -700.0, 700.0))
            alpha = 1.0 / np.sqrt(1.0 + exp_neg)
            sigma = 1.0 / np.sqrt(1.0 + exp_pos)
            eps_profile = (np.maximum(sigma, 1e-300) ** 2.0) / np.maximum(alpha, 1e-300)
        else:
            eps_profile = np.maximum(
                np.asarray(data_err_vec, dtype=np.float64).reshape(-1), 1e-300
            )
            if eps_profile.shape != lam.shape:
                raise ValueError("data_err_vec must have the same length as lam_query.")

        eps_norm = eps_profile / max(float(np.mean(eps_profile)), 1e-300)

        zeta_bar = float(self.surrogate_alpha) * drift_norm + eps_norm
        return np.nan_to_num(zeta_bar, nan=1.0, posinf=1e300, neginf=1e-300)

    def _paper_eq22_profile(self, lambda_vec_ext, data_err_vec):
        """Literal Eq. (22) profile at the N history nodes t_0, ..., t_{N-1}."""
        lam_seg = np.asarray(lambda_vec_ext[:-1], dtype=np.float64)
        return self._paper_combined_profile_on_lambda(
            lam_seg, data_err_vec=np.asarray(data_err_vec, dtype=np.float64)
        )

    def sel_lambdas_lof_obj(self, lambda_vec, eps):
        """
        Objective function for optimal timestep selection using local order of approximation.
        Uses the additive normalized empirical surrogate in manuscript Eq. (22).
        """
        lambda_eps, lambda_T = self.lambda_func(eps).item(), self.lambda_func(self.T).item()
        lambda_vec_ext = np.concatenate((np.array([lambda_T]), lambda_vec, np.array([lambda_eps])))
        N = len(lambda_vec_ext) - 1

        hv = np.zeros(N)
        for i in range(N):
            hv[i] = lambda_vec_ext[i+1] - lambda_vec_ext[i]
        elv = np.exp(lambda_vec_ext)

        # Manuscript training-error profile: tilde-epsilon_t = epsilon_t * sigma_t^2 / alpha_t.
        # The paper-matched implementation uses epsilon_t == 1, hence
        # tilde-epsilon_t = sigma_t^2 / alpha_t.
        # Eq. (20) weights prediction errors at the history nodes t_0, ..., t_{N-1};
        # the terminal node t_N = eps is not a denoiser-evaluation node in the stencil.
        exp_pos = np.exp( 2.0 * lambda_vec_ext[:-1])
        exp_neg = np.exp(-2.0 * lambda_vec_ext[:-1])
        alpha_vec = 1.0 / np.sqrt(1.0 + exp_neg)
        sigma_vec = 1.0 / np.sqrt(1.0 + exp_pos)
        data_err_vec = (np.maximum(sigma_vec, 1e-300) ** 2.0) / np.maximum(alpha_vec, 1e-300)

        # Paper Eq. (22): normalized drift profile + normalized training-error profile.

        eff_err = self._paper_eq22_profile(lambda_vec_ext, data_err_vec)

        truncNum = 0
        res = 0.0
        c_vec = np.zeros(N)
        for s in range(N):
            if s in [0, N-1]:
                n = s
                J0 = elv[n+1] - elv[n]
                res += abs(J0 * eff_err[n])
            elif s in [1, N-2]:
                n = s-1
                J0 = -elv[n+1] * self.H1(hv[n+1]) / max(hv[n], 1e-8)
                J1 =  elv[n+1] * (self.H1(hv[n+1]) + hv[n]*self.H0(hv[n+1])) / max(hv[n], 1e-8)
                if s >= truncNum:
                    c_vec[n]   += eff_err[n]   * J0
                    c_vec[n+1] += eff_err[n+1] * J1
                else:
                    res += np.sqrt((eff_err[n]   * J0)**2 + (eff_err[n+1] * J1)**2)
            else:
                n = s-2
                d0 = max(hv[n], 1e-8)
                d1 = max(hv[n+1], 1e-8)
                J0 =  elv[n+2] * (self.H2(hv[n+2]) + hv[n+1]*self.H1(hv[n+2])) / (d0*(d0+hv[n+1]))
                J1 = -elv[n+2] * (self.H2(hv[n+2]) + (hv[n]+hv[n+1])*self.H1(hv[n+2])) / (d0*d1)
                J2 =  elv[n+2] * (self.H2(hv[n+2]) + (2*hv[n+1]+hv[n])*self.H1(hv[n+2]) + hv[n+1]*(hv[n]+hv[n+1])*self.H0(hv[n+2])) / (d1*(d0+hv[n+1]))
                if s >= truncNum:
                    c_vec[n]   += eff_err[n]   * J0
                    c_vec[n+1] += eff_err[n+1] * J1
                    c_vec[n+2] += eff_err[n+2] * J2
                else:
                    res += np.sqrt((eff_err[n]   * J0)**2 + (eff_err[n+1] * J1)**2 + (eff_err[n+2] * J2)**2)
        res += np.sum(np.abs(c_vec))
        return float(res)


    def get_ts_lambdas(self, N, eps, initType):
        """
        Generate N timesteps for diffusion sampling.
        
        Args:
            N: Number of timesteps
            eps: Starting time t_0 of diffusion sampling (e.g., 1e-3 for VP models)
            initType: Initialization type
                - Types with '_origin': baseline discretizations without optimization
                - Types without '_origin': optimized discretizations
                - 'unif_t': Recommended for latent-space diffusion models
                - 'unif': Recommended for pixel-space diffusion models (logSNR initialization)
                - 'phi': phi-domain mass-uniform initialization
        
        Returns:
            t_res: Timesteps in time domain
            lambda_res: Timesteps in log-SNR domain
        """

        lambda_eps, lambda_T = self.lambda_func(eps).item(), self.lambda_func(self.T).item()

        # Optimization constraints: ensure lambda_T < lambda_1 < ... < lambda_N < lambda_eps.
        constr_mat = np.zeros((N, N-1)) 
        for i in range(N-1):
            constr_mat[i][i] = 1.
            constr_mat[i+1][i] = -1
        lb_vec = np.zeros(N)
        lb_vec[0], lb_vec[-1] = lambda_T, -lambda_eps

        ub_vec = np.zeros(N)
        for i in range(N):
            ub_vec[i] = np.inf
        linear_constraint = LinearConstraint(constr_mat, lb_vec, ub_vec)

        # Initialize timestep vector based on selected strategy
        if initType in ['unif', 'unif_origin']:
            lambda_vec_ext = torch.linspace(lambda_T, lambda_eps, N+1)
        elif initType in ['unif_t', 'unif_t_origin']:
            t_vec = torch.linspace(self.T, eps, N+1)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['edm', 'edm_origin']:
            rho = 7
            edm_sigma_min, edm_sigma_max = self.edm_sigma(eps).item(), self.edm_sigma(self.T).item()
            edm_sigma_vec = torch.linspace(edm_sigma_max**(1. / rho), edm_sigma_min**(1. / rho), N + 1).pow(rho)
            t_vec = self.edm_inverse_sigma(edm_sigma_vec)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['quad', 'quad_origin']:
            t_order = 2
            t_vec = torch.linspace(self.T**(1./t_order), eps**(1./t_order), N+1).pow(t_order)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['phi', 'phi_origin']:
            # ================================================================
            # DU + PS COUPLED INITIALIZATION (paper-matched two-stage pipeline)
            # ================================================================
            # DU does NOT use an independent sigma/alpha-only profile here.
            # Instead it uses the SAME normalized empirical profile zeta_bar(lambda)
            # as the PS refinement objective:
            #
            #   zeta_bar = alpha_PS * Norm[Lhat * C_path] + Norm[eps_tilde].
            #
            # DU then forms
            #   g(lambda)   = zeta_bar(lambda) * exp(lambda),
            #   phi(lambda) = integral_{lambda_T}^{lambda} g(u) du,
            # and chooses equal-mass nodes in phi.  These DU nodes are subsequently
            # refined by the constrained PS objective below when initType='phi'.
            # For initType='phi_origin', only the DU initialization is returned.

            lambda_eps = self.lambda_func(eps).item()
            lambda_T   = self.lambda_func(self.T).item()

            dense_M = max(int(self.paper_cpath_dense_M), 4096)
            lam_dense = np.linspace(lambda_T, lambda_eps, dense_M, dtype=np.float64)

            # Same Eq. (22) temporal profile used by the PS objective.
            zeta_dense = self._paper_combined_profile_on_lambda(lam_dense)

            # Denoising-Uniform density and cumulative difficulty coordinate.
            g = np.maximum(zeta_dense, 1e-300) * np.exp(np.clip(lam_dense, -700.0, 700.0))
            dlam = np.diff(lam_dense)
            integ = 0.5 * (g[:-1] + g[1:]) * dlam
            phi_dense = np.concatenate([[0.0], np.cumsum(integ)])

            if not np.all(np.isfinite(phi_dense)) or phi_dense[-1] <= phi_dense[0]:
                raise FloatingPointError(
                    "Invalid DU cumulative profile. Check Ltheta calibration and noise schedule."
                )

            # Equal difficulty mass: phi(lambda_s) = s/N * phi(lambda_eps).
            phi_targets = np.linspace(phi_dense[0], phi_dense[-1], N + 1, dtype=np.float64)
            lam_nodes = np.interp(phi_targets, phi_dense, lam_dense)

            # Exact endpoints and strict ordering.
            lam_nodes[0], lam_nodes[-1] = lambda_T, lambda_eps
            eps_push = 1e-12 * max(lambda_eps - lambda_T, 1.0)
            for i in range(1, lam_nodes.shape[0]):
                if not (lam_nodes[i] > lam_nodes[i - 1]):
                    lam_nodes[i] = lam_nodes[i - 1] + eps_push
            lam_nodes[-1] = lambda_eps

            lambda_vec_ext = torch.tensor(lam_nodes, dtype=torch.float64)

        else:
            print('InitType not found!')
            return 

        if initType in ['unif_origin', 'unif_t_origin', 'edm_origin', 'quad_origin','phi_origin']:
                lambda_res = lambda_vec_ext
                t_res = torch.tensor(self.inverse_lambda(lambda_res))
        else:
            # Optimize timesteps using trust-constr method
            lambda_vec_init = np.array(lambda_vec_ext[1:-1])
            res = minimize(
                self.sel_lambdas_lof_obj, 
                lambda_vec_init, 
                method='trust-constr', 
                args=(eps,), 
                constraints=[linear_constraint], 
                options={'verbose': self.optimizer_verbose}
            )
            lambda_res = torch.tensor(np.concatenate((
                np.array([lambda_T]), res.x, np.array([lambda_eps])
            )))
            t_res = torch.tensor(self.inverse_lambda(lambda_res))
        
        return t_res, lambda_res

    def schedule_dict(self, N, eps, initType, *, meta=None):
        """Return a generated schedule as a JSON-serializable dictionary."""
        t_res, lambda_res = self.get_ts_lambdas(N, eps, initType)
        payload = {
            "nfe": int(N),
            "eps": float(eps),
            "init_type": initType,
            "t": [float(x) for x in t_res.detach().cpu().flatten().tolist()],
            "lambda": [float(x) for x in lambda_res.detach().cpu().flatten().tolist()],
            "parameters": {
                "objective": "paper_eq22_additive_normalized",
                "l_theta": None if self.L_theta is None else float(self.L_theta),
                "ltheta_mode": self.Ltheta_mode,
                "surrogate_alpha": float(self.surrogate_alpha),
                "paper_cpath_dense_M": int(self.paper_cpath_dense_M),
                "training_error_profile": "sigma^2/alpha (epsilon_t=1)",
                "du_profile": "g(lambda)=zeta_bar(lambda)*exp(lambda); zeta_bar follows Eq.(22)",
            },
        }
        if meta:
            payload["meta"] = dict(meta)
        return payload

    def save_schedule(self, path, N, eps, initType, *, meta=None):
        """Generate and save a schedule JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.schedule_dict(N, eps, initType, meta=meta)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
