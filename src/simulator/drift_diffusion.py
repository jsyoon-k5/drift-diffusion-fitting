"""
Drift Diffusion Model Simulation

Recommended parameter range:
a: [0.045, 0.18]
mu: [0.00, 0.50]
T_er: [0.05, 0.40]

Code written by June-Seop Yoon
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Union, Optional
from box import Box
from matplotlib.lines import Line2D
import os
from pathlib import Path
import pandas as pd

from ..config.config import CFG_DRDF, SYMBOLS
from ..utils.myplot import figure_grid, figure_save
from ..utils.myutils import get_compact_timestamp_str


ROOT_DIR = Path(__file__).parent.parent.parent
FOLDERNAMES = {
    "hist": "simul_hist",
    "single": "single_simul" 
}


class DriftDiffusionSimulator:
    def __init__(self, config: Union[dict, Box, str]="default"):
        """
        Initialize the Drift Diffusion Simulator with parameter configuration.
        config (dict): Configuration dictionary specifying parameter ranges or fixed values.
        If a string (default="default"), it will use the corresponding configuration from CFG_DRDF.
        If a Box (or dict) object, it will use it directly.

        There are nine parameters in total:
        [a, mu, T_er, eta, s_z, s_t, sigma, step_size, a_z_ratio]
        config variable should include all nine parameters.
        """
        if isinstance(config, str):
            config = CFG_DRDF[config]
        elif isinstance(config, dict):
            config = Box(config)

        self.param_variables = dict()
        self.param_fixed = dict()

        # Check configuration and set parameters
        for param in ["a", "mu", "T_er", "eta", "s_z", "s_t", "sigma", "step_size", "a_z_ratio"]:
            assert param in config, f"Parameter '{param}' not found in the configuration."

            if isinstance(config[param], list):
                assert len(config[param]) == 2, f"Parameter range for '{param}' should be a list of two elements [min, max]."
                self.param_variables[param] = np.array(config[param])
            else:
                assert isinstance(config[param], (int, float)), f"Fixed parameter '{param}' should be a single numeric value."
                self.param_fixed[param] = config[param]
        
        self.config = config
    
    
    def get_config(self):
        return self.config
    

    def adjustable_parameters(self):
        """
        Return the list of adjustable (variable) parameter names.
        """
        l = list(self.param_variables.keys())
        l.sort()
        return l, self.param_variables

    
    def run_simulation(
        self,
        params: Optional[Union[dict, Box]] = None,
        num_of_simul: int = 1000,
        verbose: bool = False,
        return_params: bool = False
    ):
        """
        Run multiple trials of drift diffusion simulation
        with given parameters.
        params (dict): Dictionary containing parameter values.
        num_of_simul (int): Number of simulation trials to run.
        Returns reaction time and correct/error result in numpy array.
        """
        if params is not None:
            assert set(params.keys()) == set(self.param_variables.keys()), \
                f"Parameter dictionary 'params' must include all adjustable parameters: {list(self.param_variables.keys())}."
            _params = params.copy()
        else:
            _params = self.sample_parameters()
        _params.update(self.param_fixed)    

        reaction_time, correction = list(), list()

        for _ in (tqdm(range(num_of_simul), desc="Drift Diffusion Simulation") if verbose else range(num_of_simul)):
            rt, corr, _, _ = self._single_simulation(_params, _log_evidence=False)
            reaction_time.append(rt)
            correction.append(corr)
        
        df = pd.DataFrame({"reaction_time": reaction_time, "correct": correction})

        if return_params:
            return df, _params
        return df
    

    def visualize_simulation_histogram(
        self,
        params: Optional[Union[dict, Box]] = None,
        num_of_simul: int = 1000,
        verbose: bool = False,
        save_image: bool = False,
        save_filename: Optional[str] = None,
    ):
        """
        Visualize histogram of reaction times from multiple drift diffusion simulations.
        """
        assert num_of_simul > 10, "Number of simulations must be greater than 10 for histogram visualization."
        df, _params = self.run_simulation(
            params=params,
            num_of_simul=num_of_simul,
            verbose=verbose,
            return_params=True
        )
        rt, cor = df["reaction_time"].to_numpy(), df["correct"].to_numpy()
        fig, ax = figure_grid(1, 1, size_ax=np.array([8, 5]))
        ax.hist(rt[cor==1], bins=40, density=False, histtype='step', label=f'Correct = {np.sum(cor==1)} ({np.sum(cor==1)/len(cor)*100:.2f} %)')
        ax.hist(rt[cor==0], bins=40, density=False, histtype='step', label=f'Error = {np.sum(cor==0)} ({np.sum(cor==0)/len(cor)*100:.2f} %)')
        ax.axvline(x=np.mean(rt), color='blue', linestyle='--', label=f'Mean RT = {np.mean(rt):.3f}')
        ax.set_title(f"Drift Diffusion Simulation Histogram (n={num_of_simul})")
        ax.legend()
        ax.set_xlabel("Reaction Time (s)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

        self.__add_figure_legend_parameters(fig, _params)

        if not save_image:
            plt.show()
        else:
            if save_filename is None:
                save_filename = get_compact_timestamp_str()
            path = os.path.join(ROOT_DIR, f"data/visualization/{FOLDERNAMES['hist']}/{save_filename}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            figure_save(fig, path, dpi=100)
            if verbose:
                print(f"Figure saved to: {path}")



    def visualize_single_simulation(
        self, 
        params: Optional[Union[dict, Box]] = None,
        save_image: bool = False,
        save_filename: Optional[str] = None,
        verbose: bool = False,
    ):
        _params = self.sample_parameters() if params is None else params.copy()
        _params.update(self.param_fixed)

        rt, correct, evidence_trace, timesteps = self._single_simulation(_params, _log_evidence=True)

        if verbose:
            print(f"Parameters used for simulation: {_params}")
            print(f"Reaction time: {rt}, Correct: {correct}")

        fig, ax = figure_grid(1, 1, size_ax=np.array([10, 4]))
        ax.step(timesteps, evidence_trace, where='post', label='Evidence')
        ax.axhline(y=_params['a'], color='red', linestyle='--', label='Decision boundary')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title(f"Drift Diffusion Single Trial Simulation (RT={rt:.3f}, Correct={'Yes' if correct==1 else 'No'})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amount of\naccumulated evidence")

        ax.set_xlim(0, timesteps[-1] + _params["step_size"])
        ax.legend()
        ax.grid(alpha=0.3)
        
        self.__add_figure_legend_parameters(fig, _params)

        if not save_image:
            plt.show()
        else:
            if save_filename is None:
                save_filename = get_compact_timestamp_str()
            path = os.path.join(ROOT_DIR, f"data/visualization/{FOLDERNAMES['single']}/{save_filename}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            figure_save(fig, path, dpi=100)
            if verbose:
                print(f"Figure saved to: {path}")


    def sample_parameters(self, fill_fixed: bool=False):
        """
        Sample parameters from the defined variable ranges.
        Note that fixed parameters are not included in the sampled output.
        """
        sampled_params = dict()
        for param, (min_val, max_val) in self.param_variables.items():
            sampled_params[param] = np.random.uniform(min_val, max_val)
        if fill_fixed:
            sampled_params.update(self.param_fixed)
        return sampled_params
    

    def _single_simulation(self, params, _max_init_cond_sample=10000, _log_evidence=False):
        """
        Perform a single trial of drift diffusion simulation
        with given parameters.
        params (dict): Dictionary containing parameter values.
        Returns reaction time and correct/error result.
        """
        a = params['a']
        mu = params['mu']
        T_er = params['T_er']
        eta = params['eta']
        s_z = params['s_z']
        s_t = params['s_t']
        sigma = params['sigma']
        step_size = params['step_size']
        a_z_ratio = params['a_z_ratio']
        z = a / a_z_ratio

        sample_success = False
        for _ in range(_max_init_cond_sample):
            sample_mu = np.random.normal(mu, eta)
            sample_T_er = T_er + np.random.uniform(-0.5, 0.5) * s_t
            evidence = z + np.random.uniform(-0.5, 0.5) * s_z

            if sample_mu > 0 and sample_T_er > 0 and evidence > 0 and evidence < a:
                sample_success = True
                break
        
        if not sample_success:
            raise ValueError(f"Reached the maximum initial condition sample attemption ({_max_init_cond_sample}):\nCheck parameter ranges and variability settings.")
        
        p = 0.5 * (1 + sample_mu * np.sqrt(step_size) / sigma)
        delta = sigma * np.sqrt(step_size)

        num_step = 0
        evidence_trace = [evidence] if _log_evidence else None

        while True:
            if np.random.binomial(1, p):
                evidence += delta
            else:
                evidence -= delta

            if _log_evidence:
                evidence_trace.append(evidence)
            num_step += 1

            if evidence >= a or evidence <= 0:
                rt = num_step * step_size + sample_T_er
                correct = 1 if evidence >= a else 0
                break
        
        if _log_evidence:
            evidence_trace = np.array(evidence_trace)
            timesteps = np.arange(len(evidence_trace)) * step_size # + sample_T_er
        else:
            timesteps = None
        
        return rt, correct, evidence_trace, timesteps
    

    def __add_figure_legend_parameters(self, fig: plt.Figure, _params):
        # Create parameter legend as figure legend on the right
        variable_labels = [f"{SYMBOLS[k]} = {_params[k]:.5f}" for k in self.param_variables.keys()]
        fixed_labels = [f"{SYMBOLS[k]} = {_params[k]:.5f}" for k in self.param_fixed.keys()]
        
        # Add empty label for spacing between variable and fixed params
        all_labels = variable_labels + [''] + fixed_labels
        handles = [Line2D([0], [0], marker='', linestyle='', label=label) for label in all_labels]
        
        # Create figure legend on the right side
        # Use outside='right' with constrained_layout
        fig.legend(handles=handles, loc='outside right center',
                   title='Parameters', frameon=True, 
                   handlelength=0, handletextpad=0)




if __name__ == "__main__":
    """Example code for drift diffusion simulation visualization."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="default")
    parser.add_argument('--num_simul', type=int, default=1000)
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--verbose', type=bool, default=True)

    args = parser.parse_args()

    simulator = DriftDiffusionSimulator(config=args.config)
    
    if args.num_simul > 1:
        simulator.visualize_simulation_histogram(
            num_of_simul=args.num_simul,
            verbose=args.verbose,
            save_image=args.save_image,
            save_filename=args.filename
        )
    else:
        simulator.visualize_single_simulation(
            verbose=args.verbose,
            save_image=args.save_image,
            save_filename=args.filename
        )



