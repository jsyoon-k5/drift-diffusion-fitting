"""
Maximum likelihood estimation for drift diffusion model parameters.
Code written by June-Seop Yoon
Pymoo implemented by Namsub Kim
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping
from typing import Optional, Dict, Tuple, List
import pandas as pd
import os
from pathlib import Path

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize

from src.simulator.drift_diffusion import DriftDiffusionSimulator
from src.utils.mymath import ecdf
from src.datakit.loader import get_user_data


def drift_diffusion_likelihood(
    simulated_rt: np.ndarray,
    simulated_cor: np.ndarray,
    empirical_rt: np.ndarray,
    empirical_cor: np.ndarray,
    dt: float = 0.01,
    negation: bool = False,
    pad_max_rt: float = 5.0
):
    """
    Calculate the likelihood of empirical data given simulated data
    from drift diffusion model.

    simulated_rt (ndarray): Simulated reaction times.
    simulated_cor (ndarray): Simulated correctness (1 for correct, 0 for error).
    empirical_rt (ndarray): Empirical reaction times.
    empirical_cor (ndarray): Empirical correctness (1 for correct, 0 for error).
    dt (float): Time step for likelihood calculation.
    negation (bool): If True, return negative log-likelihood (for optimization purpose).
    ---
    outputs (float): Log-likelihood of the empirical data given the simulated data.
    """
    likelihood = 0.0
    count = 0

    for cor_value in [1, 0]:
        simul_rt = simulated_rt[simulated_cor == cor_value]
        emp_rt = empirical_rt[empirical_cor == cor_value]

        if simul_rt.size == 0 or emp_rt.size == 0:
            continue
        
        # Compute ECDF for simulated reaction times
        rt_values, cdf_values = ecdf(simul_rt)
        # Padding for extreme values
        left_pad, right_pad = np.arange(-1, rt_values[0], dt), np.arange(rt_values[-1] + dt, pad_max_rt, dt)
        rt_values = np.concatenate((left_pad, rt_values, right_pad))
        cdf_values = np.concatenate((np.zeros(left_pad.shape), cdf_values, np.ones(right_pad.shape)))
        prob_cor = simul_rt.size / simulated_rt.size
        cdf_values *= prob_cor

        # Calculate likelihood for empirical reaction times
        cdf_t1 = np.interp(emp_rt, rt_values, cdf_values)
        cdf_t2 = np.interp(emp_rt + dt, rt_values, cdf_values)
        rt_ll = (cdf_t2 - cdf_t1) / dt
        rt_ll[rt_ll <= 0] = 1e-10  # Small float to avoid log(0)
        likelihood += np.sum(np.log(rt_ll))
        count += emp_rt.size

    return likelihood / count if not negation else -likelihood / count


def ddm_maximum_likelihood_estimation_scipy(
    empirical_rt: np.ndarray,
    empirical_cor: np.ndarray,
    simul_config: str = "default",
    verbose: int = 2,
    simul_num: Optional[int] = None,
    kwargs={}
):
    """
    Perform maximum likelihood estimation for drift diffusion model parameters
    based on empirical reaction time and correctness data.

    empirical_rt (ndarray): Empirical reaction times.
    empirical_cor (ndarray): Empirical correctness (1 for correct, 0 for error).
    ---
    outputs (dict): Dictionary containing estimated parameters and log-likelihood.
    """
    simulator = DriftDiffusionSimulator(config=simul_config)
    fit_param, fit_param_range = simulator.adjustable_parameters()

    def neg_log_likelihood(params):
        df = simulator.run_simulation(
            params=dict(zip(fit_param, params)),
            num_of_simul=empirical_rt.shape[0] if simul_num is None else simul_num,
            verbose=False,
            return_params=False
        )
        simul_rt, simul_cor = df["reaction_time"].to_numpy(), df["correct"].to_numpy()

        log_likelihood = drift_diffusion_likelihood(
            simul_rt,
            simul_cor,
            empirical_rt,
            empirical_cor,
            negation=True
        )
        return log_likelihood

    opt_kwargs = {
        'method': 'Nelder-Mead',
        # 'disp': verbose,
        'maxiter': 100,
        # 'maxfev': 10000,
        'xatol': 1e-3,
        'fatol': 2.5,
        'adaptive': True  
    }
    opt_kwargs.update(kwargs)

    initial_params = [np.mean(fit_param_range[p]) for p in fit_param]  # Initial guesses for a, mu, T_er
    bounds = [fit_param_range[p] for p in fit_param]  # Bounds for parameters

    # Display optimization progress
    if verbose == 1:
        iteration_count = [0]
        def callback(xk):
            iteration_count[0] += 1
            new_xk = list(xk)
            new_xk.append(iteration_count[0])
            obj_val = neg_log_likelihood(new_xk)
            print(f"Iter {iteration_count[0]}: a={xk[0]:.4f}, mu={xk[1]:.4f}, T_er={xk[2]:.4f}, f={-obj_val:.2f}")

    elif verbose >= 2:
        pbar = tqdm(total=opt_kwargs["maxiter"], desc="Optimization")

        def callback(xk):
            pbar.update(1)
            postfix_dict = {p: f'{xk[i]:.3f}' for i, p in enumerate(fit_param)}
            pbar.set_postfix(postfix_dict)


    result = minimize(neg_log_likelihood, initial_params, bounds=bounds, 
                      method=opt_kwargs.pop('method'), options=opt_kwargs,
                      callback=callback if verbose else None)

    if verbose >= 2:
        pbar.update(1)
        pbar.close()

    estimated_params = {param: result.x[i] for i, param in enumerate(fit_param)}
    print(estimated_params)
    return estimated_params, result.fun


class DriftDiffusionMLEProblem(Problem):
    def __init__(
        self, 
        simulator: DriftDiffusionSimulator, 
        empirical_rt: np.ndarray, 
        empirical_cor: np.ndarray,
        simul_num: Optional[int] = None
    ):
        self.simulator = simulator
        self.empirical_rt = empirical_rt
        self.empirical_cor = empirical_cor
        
        self.simul_num = simul_num
        n_obj = 1

        fit_param, fit_param_range = simulator.adjustable_parameters()
        self.fit_param = fit_param
        n_var = len(fit_param)

        xl = np.array([fit_param_range[p][0] for p in fit_param])
        xu = np.array([fit_param_range[p][1] for p in fit_param])

        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, type_var=np.float64)

    def _evaluate(self, X: np.ndarray, out: Dict, *args, **kwargs):
        F = np.zeros((X.shape[0], self.n_obj))
        n_simul = self.empirical_rt.shape[0] if self.simul_num is None else self.simul_num

        for i, params_array in enumerate(X):
            params_dict = dict(zip(self.fit_param, params_array))

            df: pd.DataFrame = self.simulator.run_simulation(
                params=params_dict,
                num_of_simul=n_simul,
                verbose=False,
                return_params=False
            )
            simul_rt, simul_cor = df["reaction_time"].to_numpy(), df["correct"].to_numpy()

            neg_log_likelihood = drift_diffusion_likelihood(
                simul_rt, simul_cor,
                self.empirical_rt, self.empirical_cor,
                negation=True
            )

            F[i, 0] = neg_log_likelihood

        out["F"] = F


def ddm_maximum_likelihood_estimation_pymoo(
    empirical_rt: np.ndarray,
    empirical_cor: np.ndarray,
    simul_config="default",
    simul_num: Optional[int] = None,
    verbose=True,
    kwargs: Dict = {}
) -> Tuple[Dict[str, float], float]:
    
    simulator = DriftDiffusionSimulator(config=simul_config)
    fit_param, _ = simulator.adjustable_parameters()

    problem = DriftDiffusionMLEProblem(
        simulator=simulator,
        empirical_rt=empirical_rt,
        empirical_cor=empirical_cor,
        simul_num=simul_num
    )

    ga_kwargs = {'pop_size': 10, 'n_gen': 20}
    ga_kwargs.update(kwargs)

    algorithm = GA(
        pop_size=ga_kwargs.pop('pop_size'),
        eliminate_duplicates=True
    )

    result = pymoo_minimize(
        problem,
        algorithm,
        termination=('n_gen', ga_kwargs.pop('n_gen')),
        verbose=verbose,
        save_history=False,
        **ga_kwargs
    )

    if result.X is None:
        raise RuntimeError("PyMoo optimization failed to find a solution.")

    estimated_params = {param: result.X[i] for i, param in enumerate(fit_param)}
    return estimated_params, result.F[0]



def fit_user_data_mle(
    user_id: int, 
    mode: str = 'scipy',
    simul_config: str = "default",
    verbose: int = 2, 
    simul_num: Optional[int] = None, 
    kwargs={}
):
    """
    Fit drift diffusion model to user data using maximum likelihood estimation.

    user_id (int): ID of the user to fit the model for.
    ---
    outputs (dict): Dictionary containing estimated parameters and log-likelihood.
    """
    simulator = DriftDiffusionSimulator(config=simul_config)
    user_df = get_user_data(user_id)
    empirical_rt, empirical_cor = user_df["reaction_time"].to_numpy(), user_df["correct"].to_numpy()

    if mode == 'scipy':
        estimated_params, _ = ddm_maximum_likelihood_estimation_scipy(
            empirical_rt, empirical_cor, simul_config=simul_config,
            verbose=verbose, simul_num=simul_num, kwargs=kwargs)
        simul_df = simulator.run_simulation(
            params=estimated_params,
            num_of_simul=empirical_rt.shape[0],
            verbose=False,
            return_params=False
        )
        simul_rt, simul_cor = simul_df["reaction_time"].to_numpy(), simul_df["correct"].to_numpy()

    elif mode == 'pymoo':
        estimated_params, _ = ddm_maximum_likelihood_estimation_pymoo(
            empirical_rt, empirical_cor, simul_config=simul_config,
            simul_num=simul_num, kwargs=kwargs)
        simul_df = simulator.run_simulation(
            params=estimated_params,
            num_of_simul=empirical_rt.shape[0],
            verbose=False,
            return_params=False
        )
        simul_rt, simul_cor = simul_df["reaction_time"].to_numpy(), simul_df["correct"].to_numpy()
    else:
        raise NotImplementedError(f"MLE mode '{mode}' is not implemented.")
    
    return estimated_params, empirical_rt, empirical_cor, simul_rt, simul_cor

if __name__ == "__main__":
    fit_user_data_mle(0, verbose=1, mode="pymoo")