"""
Approximate Bayesian Computation for drift diffusion model parameters.
Code originally written by Jonghyun Kim, modified by June-Seop Yoon
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional
from datetime import datetime
from joblib import Parallel, delayed
import psutil
import os
from pathlib import Path

from ..utils.myplot import figure_grid, figure_save
from ..simulator.drift_diffusion import DriftDiffusionSimulator
from ..datakit.loader import get_user_data
from ..config.config import SYMBOLS

MAX_CPU_CORE = psutil.cpu_count(logical=False)
DIR_TO_DATA = Path(__file__).parent.parent.parent


def ddm_approximate_bayesian_computation(
    empirical_rt: np.ndarray,
    empirical_cor: np.ndarray,
    simul_config: str = "default",
    verbose: int = 2,
    n_jobs: int = 8,
    simul_num: Optional[int] = None,
    n_samples: int = 5000,
    epsilon: Optional[float] = None,
    epsilon_quantile: float = 0.1,
    use_error_only: bool = False,
):
    """
    Approximate Bayesian Computation (simple rejection) for DDM parameters.

    Parameters
    ----------
    simulator : DriftDiffusionSimulator
        Configured simulator instance.

    empirical_rt : np.ndarray
        Empirical reaction times.

    empirical_cor : np.ndarray
        Empirical correctness (1 for correct, 0 for error).

    verbose : int
        0: quiet, 1: simple logs, 2: tqdm progress bar.

    simul_num : int or None
        Number of simulated trials per parameter sample.
        If None, use empirical_rt.shape[0].

    n_samples : int
        Number of prior samples to draw.

    epsilon : float or None
        Acceptance threshold. If None, it will be set as the
        epsilon_quantile of distances.

    epsilon_quantile : float
        Quantile used to determine epsilon when epsilon is None.
        (e.g., 0.1 -> best 10% accepted)

    use_error_only : bool
        If True, use only error rate as summary.
        If False, use [mean_rt, error_rate].

    Returns
    -------
    estimated_params : dict
        {
            **{params: posterior mean estimates},
            "epsilon": epsilon actually used,
            "n_accepted": number of accepted samples,
            "accepted_params": (n_accepted, n_param) array
        }
    """
    simulator = DriftDiffusionSimulator(config=simul_config)
    fit_param, fit_param_range = simulator.adjustable_parameters()

    # ----- 0. empirical summary -----
    emp_rt = empirical_rt
    emp_cor = empirical_cor

    emp_mean_rt = np.mean(emp_rt)
    emp_err_rate = 1.0 - np.mean(emp_cor)  # error rate

    if use_error_only:
        emp_summary = np.array([emp_err_rate], dtype=float)
    else:
        emp_summary = np.array([emp_mean_rt, emp_err_rate], dtype=float)


    # def distance(sim_summary: np.ndarray) -> float:
    #     # L2(Euclidean) distance between empirical & simulated summaries
    #     return float(np.linalg.norm(emp_summary - sim_summary))

    # ----- 1. prior ranges from simulator -----
    all_params = np.zeros((n_samples, len(fit_param)), dtype=float)
    all_distances = np.zeros(n_samples, dtype=float)

    def worker(job_id):
        np.random.seed(datetime.now().microsecond + job_id)
        df, sample_params = simulator.run_simulation(
            params=None,
            num_of_simul=emp_rt.shape[0] if simul_num is None else simul_num,
            verbose=False,
            return_params=True,
        )
        sim_rt, sim_cor = df["reaction_time"].to_numpy(), df["correct"].to_numpy()
        sim_mean_rt = np.mean(sim_rt)
        sim_err_rate = 1.0 - np.mean(sim_cor)

        if use_error_only:
            sim_summary = np.array([sim_err_rate], dtype=float)
        else:
            sim_summary = np.array([sim_mean_rt, sim_err_rate], dtype=float)

        dist = float(np.linalg.norm(emp_summary - sim_summary))
        param_values = np.array([sample_params[p] for p in fit_param], dtype=float)
        return param_values, dist
    
    res = Parallel(n_jobs=min(MAX_CPU_CORE, n_jobs))(
        delayed(worker)(job_id) for job_id in (range(n_samples) if verbose < 2 else \
                                               tqdm(range(n_samples), desc="ABC (simple rejection)"))
    )
    for i in range(n_samples):
        all_params[i], all_distances[i] = res[i]


    # ----- 2. epsilon 결정 -----
    if epsilon is None:
        epsilon = np.quantile(all_distances, epsilon_quantile)

    # ----- 3. accept / reject -----
    accept_mask = (all_distances <= epsilon) & np.isfinite(all_distances)
    accepted_params = all_params[accept_mask, :]

    if accepted_params.shape[0] == 0:
        raise RuntimeError(
            f"No samples accepted in ABC. "
            f"Try increasing epsilon or epsilon_quantile (current epsilon={epsilon:.4f})."
        )

    # posterior mean as point estimate
    post_mean = np.mean(accepted_params, axis=0)

    estimated_params = {
        **{p: float(post_mean[i]) for i, p in enumerate(fit_param)},
        "epsilon": float(epsilon),
        "n_accepted": int(accepted_params.shape[0]),
        "accepted_params": accepted_params,
        "acc_rate": float(accepted_params.shape[0] / n_samples),
    }

    if verbose >= 1:
        acc_rate = accepted_params.shape[0] / n_samples
        print(
            f"ABC finished. epsilon={epsilon:.4f}, "
            f"accepted={accepted_params.shape[0]}/{n_samples} "
            f"({acc_rate*100:.2f}% acceptance rate)"
        )
        print("Posterior means: ", end='')
        for p in fit_param:
            print(f"{p}={estimated_params[p]:.4f},")

    return estimated_params, fit_param



def visualize_posterior_histogram(
    filename: str,
    estimated_params: dict,
    fit_param: list,
    bins: int = 40
):
    """
    Visualize posterior histograms of DDM parameters (from ABC).

    Parameters
    ----------
    estimated_params : dict
        Dictionary returned by approximate_bayesian_computation, expected keys:
            - "accepted_params": ndarray of shape (n_accepted, n_param)
    """

    if "accepted_params" not in estimated_params:
        raise ValueError("estimated_params must contain key 'accepted_params'.")

    accepted_params = estimated_params["accepted_params"]  # shape: (n_accepted, n_param)

    if not isinstance(accepted_params, np.ndarray):
        accepted_params = np.array(accepted_params)

    assert accepted_params.ndim == 2 and accepted_params.shape[1] == len(fit_param), \
        f"accepted_params must be (n_samples, {len(fit_param)}) array of {fit_param}."

    # ----- figure 생성 -----
    fig, axes = figure_grid(1, 3, size_ax=np.array([5, 4]))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    def cred_interval(x, alpha=0.95):
        lower = np.quantile(x, (1 - alpha) / 2.0)
        upper = np.quantile(x, 1 - (1 - alpha) / 2.0)
        return lower, upper

    for i, (ax, samples) in enumerate(zip(axes, accepted_params.T)):
        mean_val = np.mean(samples)
        low, high = cred_interval(samples, alpha=0.95)

        ax.hist(samples, bins=bins, density=True, histtype='step', label="Posterior")
        ax.axvline(mean_val, linestyle='--', label=f"Mean = {mean_val:.3f}")
        ax.axvline(low, linestyle=':',  label=f"95% CI low = {low:.3f}")
        ax.axvline(high, linestyle=':', label=f"95% CI high = {high:.3f}")

        ax.set_title(f"Posterior of {SYMBOLS[fit_param[i]]}")
        ax.set_xlabel(SYMBOLS[fit_param[i]])
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Posterior Distributions of DDM Parameters (ABC)", fontsize=14)
    figure_save(fig, filename)


def fit_user_data_abc(
    user_id: int,
    simul_config: str = "default",
    verbose: int = 2,
    simul_num: Optional[int] = None,
    **abc_kwargs,
):
    simulator = DriftDiffusionSimulator(config=simul_config)
    user_df = get_user_data(user_id)
    empirical_rt, empirical_cor = user_df["reaction_time"].to_numpy(), user_df["correct"].to_numpy()

    estimated_params, fit_param = ddm_approximate_bayesian_computation(
        empirical_rt=empirical_rt,
        empirical_cor=empirical_cor,
        simul_config=simul_config,
        verbose=verbose,
        simul_num=simul_num,
        **abc_kwargs,
    )

    # posterior mean parameter based simulation
    simul_rt, simul_cor = simulator.run_simulation(
        params={p: estimated_params[p] for p in fit_param},
        num_of_simul=empirical_rt.shape[0],
        verbose=2,
        return_params=False,
    )

    # posterior visualization
    visualize_posterior_histogram(
        filename=os.path.join(DIR_TO_DATA, f"data/visualization/abc_posterior/user_{user_id}_posterior.png"),
        estimated_params=estimated_params, 
        fit_param=fit_param
    )

    return estimated_params, empirical_rt, empirical_cor, simul_rt, simul_cor


if __name__ == "__main__":
    fit_user_data_abc(
        user_id=0,
        simul_config="default",
        verbose=2,
        simul_num=100,
        n_samples=500,
        epsilon=0.1,
        epsilon_quantile=0.1,  # top 10 percentile
        use_error_only=False,
    )
