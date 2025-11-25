import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from matplotlib.lines import Line2D

from ..utils.myplot import figure_grid, figure_save
from ..utils.myutils import get_compact_timestamp_str, save_dict_to_yaml, unbox
from ..utils.device_info import save_device_info
from ..simulator.drift_diffusion import DriftDiffusionSimulator

from ..config.config import CFG_FITTING, SYMBOLS

from ..fitting.mle import ddm_maximum_likelihood_estimation_pymoo
from ..fitting.abc import ddm_approximate_bayesian_computation
from ..fitting.ami import DriftDiffusionInferenceEngine


from ..datakit.loader import get_user_data, N_USER

import time

# AMI model setting will standardize simulator config.
# If you want to use different simulator config, please create (train) a new InferenceEngine instance.
INF_ENGINE = DriftDiffusionInferenceEngine(model_name=CFG_FITTING["ami"]["model_name"], verbose=False)
SIMULATOR = DriftDiffusionSimulator(config=INF_ENGINE.simul_config)
FIT_PARAMS, FIT_PARAM_RANGES = SIMULATOR.adjustable_parameters()
N_STANDARD_TRIALS = INF_ENGINE.data_config.n_trial

DIR_TO_DATA = os.path.join(Path(__file__).parent.parent.parent, "data/fitting")

METHOD_NAME = {
    'mle': 'Maximum Likelihood Estimation',
    'abc': 'Approximate Bayesian Computation',
    'ami': 'Amortized Inference',
}

### Wrappers for fitting methods ###
def fit_mle(empirical_rt, empirical_cor):
    start_t = time.time()
    param, ll = ddm_maximum_likelihood_estimation_pymoo(
        empirical_rt=empirical_rt,
        empirical_cor=empirical_cor,
        simul_config=INF_ENGINE.simul_config,
        simul_num=len(empirical_rt),
        verbose=False,
        kwargs={'pop_size': CFG_FITTING["mle"]["pop_size"], 'n_gen': CFG_FITTING["mle"]["n_gen"]},
    )
    elapsed_time = time.time() - start_t
    return param, {'likelihood': ll, 'elapsed_time': elapsed_time}


def fit_abc(empirical_rt, empirical_cor):
    start_t = time.time()
    param, plist = ddm_approximate_bayesian_computation(
        empirical_rt=empirical_rt,
        empirical_cor=empirical_cor,
        simul_config=INF_ENGINE.simul_config,
        verbose=0,
        n_samples=CFG_FITTING["abc"]["n_sample"],
        n_jobs=CFG_FITTING["abc"]["n_jobs"],
        epsilon_quantile=CFG_FITTING["abc"]["epsilon_quantile"],
    )
    elapsed_time = time.time() - start_t
    return (
        {p: param[p] for p in plist}, 
        {'elapsed_time': elapsed_time, "acc_rate": param["acc_rate"]}
    )


def fit_ami(empirical_rt, empirical_cor):
    stat = np.array([empirical_rt, empirical_cor]).T
    start_t = time.time()
    _, param = INF_ENGINE.infer(stat, n_sample=300, return_dict=True, return_samples=False)
    elapsed_time = time.time() - start_t
    return param, {'elapsed_time': elapsed_time}


### Main fitting function ###
def run_fitting_single(
    mode='user',
    user_id=0,
    fitting_method='ami',
):
    # Loading data
    if mode == 'user':
        df = get_user_data(user_id)
        gt_param = None
        name = f"user_{str(user_id)}"

    elif mode == 'simul':
        df, gt_param = SIMULATOR.run_simulation(num_of_simul=N_STANDARD_TRIALS, return_params=True, verbose=False)
        gt_param = {k: gt_param[k] for k in FIT_PARAMS}
        name = f"{fitting_method}_simul_{get_compact_timestamp_str(omit_year=True)}"

    else:
        raise ValueError("mode must be 'user' or 'simul'")
    
    rt, cor = df["reaction_time"].to_numpy(), df["correct"].to_numpy()

    # Fitting
    if fitting_method == 'mle':
        estimated_params, fit_info = fit_mle(rt, cor)
    elif fitting_method == 'abc':
        estimated_params, fit_info = fit_abc(rt, cor)
    elif fitting_method == 'ami':
        estimated_params, fit_info = fit_ami(rt, cor)
    else:
        raise ValueError("fitting_method must be 'mle', 'abc', or 'ami'")
    
    # Evaluation
    estimation_df = SIMULATOR.run_simulation(
        params=estimated_params,
        num_of_simul=len(rt),
        verbose=False,
        return_params=False,
    )

    # Save results
    save_path = os.path.join(DIR_TO_DATA, f"{name}")
    os.makedirs(save_path, exist_ok=True)

    # Fitting option
    save_dict_to_yaml(unbox(CFG_FITTING[fitting_method]), os.path.join(save_path, f"{fitting_method}_fitting_config.yaml"))

    # Apparatus information
    save_device_info(os.path.join(save_path, f"{fitting_method}_device_info_log.txt"))

    # Viusalize two historgrams
    fig, ax = figure_grid(1, 1, size_ax=np.array([8, 5]))
    # Ground truth
    ax.hist(rt[cor==1], bins=10, density=False, histtype='step', color='blue', linestyle='-', label="GT. Correct")
    ax.hist(rt[cor==0], bins=10, density=False, histtype='step', color='blue', linestyle='--', label="GT. Error")
    # Estimation
    ax.hist(estimation_df["reaction_time"][estimation_df["correct"]==1], bins=10, density=False, histtype='step', color='red', linestyle='-', label="Est. Correct")
    ax.hist(estimation_df["reaction_time"][estimation_df["correct"]==0], bins=10, density=False, histtype='step', color='red', linestyle='--', label="Est. Error")
    ax.legend()
    ax.set_title(METHOD_NAME[fitting_method])
    ax.set_xlabel("Reaction Time (s)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

    
    if gt_param is not None:
        param_labels = [f"Est. {SYMBOLS[k]}={estimated_params[k]:.3f}" for k in FIT_PARAMS] + \
            [''] + [f"GT. {SYMBOLS[k]}={gt_param[k]:.3f}" for k in FIT_PARAMS]
    else:
        param_labels = [f"Est. {SYMBOLS[k]}={estimated_params[k]:.3f}" for k in FIT_PARAMS]
    handles = [Line2D([0], [0], marker='', linestyle='', label=label) for label in param_labels]

    fig.legend(handles=handles, loc='outside right center',
                title='Parameters', frameon=True, 
                handlelength=0, handletextpad=0)
    figure_save(fig, os.path.join(save_path, f"{fitting_method}_rt_histogram.png"))

    # Save dictionaries an df as csv
    param_df = pd.DataFrame([estimated_params])
    param_df.to_csv(os.path.join(save_path, f"{fitting_method}_estimated_params.csv"), index=False)
    if gt_param is not None:
        gt_param_df = pd.DataFrame([gt_param])
        gt_param_df.to_csv(os.path.join(save_path, f"{fitting_method}_ground_truth_params.csv"), index=False)
    fit_info_df = pd.DataFrame([fit_info])
    fit_info_df.to_csv(os.path.join(save_path, f"{fitting_method}_fit_info.csv"), index=False)

    behavior_df = pd.DataFrame({
        "gt_rt": rt,
        "gt_cor": cor,
        "estimation_rt": estimation_df["reaction_time"],
        "estimation_cor": estimation_df["correct"],
    })
    behavior_df.to_csv(os.path.join(save_path, f"{fitting_method}_behavior_comparison.csv"), index=False)


if __name__ == "__main__":
    run_fitting_single(
        mode='user',
        user_id=0,
        fitting_method='ami',
    )