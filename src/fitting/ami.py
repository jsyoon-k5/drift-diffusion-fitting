"""
Amortized inference for drift diffusion model parameters.
Code written by June-Seop Yoon

Original implementation: Hee-Seung Moon
(https://github.com/hsmoon121/amortized-inference-hci)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from tqdm import tqdm
from pathlib import Path
from glob import glob
from datetime import datetime
from box import Box
# import time

from joblib import Parallel, delayed
import psutil

import torch
import torch.nn.functional as F

from ..config.config import CFG_AMORT, SYMBOLS
from ..simulator.drift_diffusion import DriftDiffusionSimulator
from ..utils.mymath import linear_normalize, linear_denormalize, closest_factor_pair
from ..utils.myutils import (
    config_hash, 
    save_dict_to_yaml, 
    npz_load, 
    npz_save, 
    get_compact_timestamp_str, 
    unbox,
    load_config
)
from ..utils.myplot import figure_grid, figure_save, draw_r2_plot


from ..nets.amortizer import AmortizerForTrialData, RegressionForTrialData
from ..utils.schedulers import CosAnnealWR
from ..utils.loggers import Logger


MAX_CPU_CORE = psutil.cpu_count(logical=False)
DIR_TO_DATA = os.path.join(Path(__file__).parent.parent.parent, "data/amortizer/models")


### Dataset Classes ###
# Base Dataset Class, which can be extended for training, validation dataset.
class DriftDiffusionDataset:
    def __init__(
        self,
        simul_config='default',
        data_config='default',
    ):
        # Simulator setup
        self.simulator = DriftDiffusionSimulator(config=simul_config)
        self.fit_param, self.fit_param_range = self.simulator.adjustable_parameters()
        self.fit_param_range = np.array([self.fit_param_range[p] for p in self.fit_param])  # (N, 2)

        # Data processing setup
        if isinstance(data_config, str):
            self.data_config = CFG_AMORT['data'][data_config]
        else:   # Box or dict
            self.data_config = data_config

        self.config = unbox({
            'simul': self.simulator.get_config(),
            'data': self.data_config,
        })
        self.cfg_hash = config_hash(self.config)

        self.data_path = os.path.join(Path(__file__).parent.parent.parent, 
                                      "data/amortizer/dataset",
                                      self.cfg_hash)
        os.makedirs(self.data_path, exist_ok=True)

        # Save config
        if not os.path.exists(os.path.join(self.data_path, "config.yaml")):
            save_dict_to_yaml(self.config, os.path.join(self.data_path, "config.yaml"))

    
    def simulate(self, sim_per_param=None):
        df, params = self.simulator.run_simulation(
            params=None,
            num_of_simul=self.data_config.n_trial if sim_per_param is None else sim_per_param,
            verbose=False,
            return_params=True
        )
        df = df[self.data_config.feature].to_numpy()
        param_values = np.array([params[p] for p in self.fit_param])
        return df, param_values
    

    def load_existing_data(self, verbose=True, file_prefix='', tqdm_desc=''):
        data_files = glob(os.path.join(self.data_path, f"{file_prefix}_*.npz"))
        if len(data_files) == 0:
            raise FileNotFoundError(f"No existing data found for configuration {self.cfg_hash}.")
        
        all_params = []
        all_stat_data = []

        for file in data_files if not verbose else tqdm(data_files, desc=tqdm_desc):
            loaded = npz_load(file)
            all_params.append(loaded['params'])
            all_stat_data.append(loaded['stat_data'])
        
        all_params = np.concatenate(all_params, axis=0)     # shape: (n_param, param_dim)
        all_stat_data = np.concatenate(all_stat_data, axis=0)   # shape: (n_param, n_trial, stat_dim)

        if verbose:
            print(f"Loaded existing data from {len(data_files)} file(s) - total parameters: {all_params.shape[0]}, total trials: {all_stat_data.shape[0] * all_stat_data.shape[1]} (per parameter: {all_stat_data.shape[1]})")

        return all_params, all_stat_data


    def generate_new_data(self, n_param_per_file=64, repeat=1, n_jobs=8, file_prefix='', tag=''):
        n_jobs = min(n_jobs, MAX_CPU_CORE)
        
        print(f"Generating new {tag} data - configuration: {self.cfg_hash} (n_jobs={n_jobs})")
        print(f"Total npz files to generate: {repeat}, Total parameter sets: {n_param_per_file * repeat} ({n_param_per_file} per file), {self.data_config.n_trial} trials per parameter set.")

        def worker(job_id):
            np.random.seed(datetime.now().microsecond + job_id)
            sim_df, sim_params = self.simulate()
            normed_data = self._normalize_data(sim_df)
            normed_params = self._normalize_param(sim_params)
            return normed_params, normed_data
        
        for _ in range(repeat):
            res = Parallel(n_jobs=n_jobs)(
                delayed(worker)(job_id) for job_id in tqdm(range(n_param_per_file))
            )
            params_arr = np.array([res[i][0] for i in range(n_param_per_file)], dtype=np.float32)
            stat_arr = np.array([res[i][1] for i in range(n_param_per_file)], dtype=np.float32)

            npz_save(
                filename=os.path.join(self.data_path, f"{file_prefix}_{get_compact_timestamp_str(omit_year=True)}.npz"),
                params=params_arr,
                stat_data=stat_arr
            )

    

    @staticmethod
    def process_data(data:np.ndarray, group_size:int, aggregation_features:list):
        """
        data: (n_samples, n_features)
        Caution: data should be normalized before grouping.
        """
        n_groups = data.shape[0] // group_size + \
            (1 if data.shape[0] % group_size != 0 else 0)
        grouped_data = []
        for i in range(n_groups):
            group = data[
                i*group_size : min((i+1)*group_size, data.shape[0]),
                :
            ]
            agg_features = []
            for feature in aggregation_features:
                if feature == 'mean':
                    agg_features.append(np.mean(group, axis=0))
                elif feature == 'std':
                    agg_features.append(np.std(group, axis=0))
                elif feature == 'min':
                    agg_features.append(np.min(group, axis=0))
                elif feature == 'max':
                    agg_features.append(np.max(group, axis=0))
                elif feature == 'median':
                    agg_features.append(np.median(group, axis=0))
                else:
                    raise ValueError(f"Unknown aggregation feature: {feature}")
                # More aggregation features can be added here
            grouped_data.append(np.concatenate(agg_features))
        return np.array(grouped_data, dtype=np.float32)
    

    @staticmethod
    def process_batch_data(batch_data:np.ndarray, group_size:int, aggregation_features:list):
        batch_grouped_data = []
        for b in range(batch_data.shape[0]):
            grouped_data = DriftDiffusionDataset.process_data(
                batch_data[b],
                group_size,
                aggregation_features
            )
            batch_grouped_data.append(grouped_data)
        return np.array(batch_grouped_data, dtype=np.float32)

    

    def _normalize_param(self, param_values:np.ndarray) -> np.ndarray:
        normed_params = linear_normalize(param_values, *self.fit_param_range.T, clip=True)
        return normed_params
    
    def _denormalize_param(self, normed_params:np.ndarray) -> np.ndarray:
        param_values = linear_denormalize(normed_params, *self.fit_param_range.T)
        return param_values
    
    def _normalize_data(self, data:np.ndarray) -> np.ndarray:
        normed_data = np.array([
            linear_normalize(
                data[:, i],
                self.data_config.range[feature]['min'],
                self.data_config.range[feature]['max'],
                clip=True
            )
            for i, feature in enumerate(self.data_config.feature)
        ]).T
        return normed_data
    
    def _denormalize_data(self, normed_data:np.ndarray) -> np.ndarray:
        data = np.array([
            linear_denormalize(
                normed_data[:, i],
                self.data_config.range[feature]['min'],
                self.data_config.range[feature]['max'],
            )
            for i, feature in enumerate(self.data_config.feature)
        ]).T
        return data
    
    @staticmethod
    def _normalize_data_static(data:np.ndarray, data_config) -> np.ndarray:
        normed_data = np.array([
            linear_normalize(
                data[:, i],
                data_config.range[feature]['min'],
                data_config.range[feature]['max'],
                clip=True
            )
            for i, feature in enumerate(data_config.feature)
        ]).T
        return normed_data
    

class DriftDiffusionTrainingDataset(DriftDiffusionDataset):
    def __init__(
        self,
        simul_config='default',
        data_config='default',
        loading_existing_data=True,
        verbose=True
    ):
        super().__init__(simul_config=simul_config, data_config=data_config)
        if loading_existing_data:
            self.load_existing_data(verbose=verbose)
    

    def load_existing_data(self, verbose=True):
        all_params, all_stat_data = super().load_existing_data(
            verbose=verbose,
            file_prefix='tr',
            tqdm_desc='Loading training data'
        )

        self.n_param = all_params.shape[0]
        self.num_of_trial_per_param = all_stat_data.shape[1]

        self.dataset = dict(
            params=all_params,
            stat_data=all_stat_data
        )
        

    def generate_new_data(self, n_param_per_file=2**10, repeat=4, n_jobs=8):
        super().generate_new_data(
            n_param_per_file=n_param_per_file,
            repeat=repeat,
            n_jobs=n_jobs,
            file_prefix='tr',
            tag='training'
        )


    def sample(self, batch_sz, sim_per_param=None):
        if sim_per_param is None:
            sim_per_param = self.data_config.n_trial

        indices = np.random.choice(self.n_param, batch_sz, replace=batch_sz > self.n_param)
        ep_indices = np.random.choice(self.num_of_trial_per_param, sim_per_param)
        rows = np.repeat(indices, sim_per_param).reshape((-1, sim_per_param))
        cols = np.tile(ep_indices, (batch_sz, 1))

        return (
            self.dataset['params'][indices],
            self.dataset["stat_data"][rows, cols].squeeze(1) if sim_per_param == 1 else self.dataset["stat_data"][rows, cols]
        )


class DriftDiffusionValidationDataset(DriftDiffusionDataset):
    def __init__(
        self,
        simul_config='default',
        data_config='default',
        loading_existing_data=True,
        verbose=True
    ):
        super().__init__(simul_config=simul_config, data_config=data_config)
        if loading_existing_data:
            self.load_existing_data(verbose=verbose)
    

    def load_existing_data(self, verbose=True):
        all_params, all_stat_data = super().load_existing_data(
            verbose=verbose,
            file_prefix='vd',
            tqdm_desc='Loading validation data'
        )
        self.n_param = all_params.shape[0]
        self.dataset = dict(
            params=all_params,
            stat_data=all_stat_data
        )
    

    def generate_new_data(self, n_param=100, n_jobs=8):
        super().generate_new_data(
            n_param_per_file=n_param,
            repeat=1,
            n_jobs=n_jobs,
            file_prefix='vd',
            tag='validation'
        )

    
    def sample(self, n_user=None):
        if n_user is None:
            n_user = self.n_param
        
        user_data = list()
        params = list()

        for user in range(n_user):
            params.append(self.dataset["params"][user])
            user_data.append(self.dataset["stat_data"][user, :])

        return np.array(params, dtype=np.float32), np.array(user_data, dtype=np.float32)


### Amortized inference engine
class DriftDiffusionAmortizedInferenceTrainer:
    def __init__(
        self,
        name=None,
        load_model=None,
        train_config='default',
        simul_config='default',
        data_config='default',
    ):
        if load_model is not None:
            name = load_model
            train_config = load_config(os.path.join(DIR_TO_DATA, f"{name}/train_config.yaml"))
            simul_config = load_config(os.path.join(DIR_TO_DATA, f"{name}/simul_config.yaml"))
            data_config = load_config(os.path.join(DIR_TO_DATA, f"{name}/data_config.yaml"))
        
        self.iter = 0
        self.train_dataset = DriftDiffusionTrainingDataset(
            simul_config=simul_config,
            data_config=data_config,
        )
        
        self.val_dataset = DriftDiffusionValidationDataset(
            simul_config=simul_config,
            data_config=data_config,
        )
        self.simulator = DriftDiffusionSimulator(config=simul_config)
        self.fit_param, self.fit_param_range = self.simulator.adjustable_parameters()

        # Training configuration
        self.data_config = Box(self.train_dataset.config["data"])
        self.train_config = CFG_AMORT['engine'][train_config] if isinstance(train_config, str) \
            else train_config
        
        if load_model is None:
            self.train_config.arch.encoder.stat_sz = \
                len(self.data_config.feature) if not self.train_config.data_grouping.enabled else \
                len(self.data_config.feature) * len(self.train_config.data_grouping.aggregation_features)
            
            if self.train_config.point_estimation:
                self.train_config.arch.linear.out_sz = len(self.fit_param)
                self.train_config.arch.linear.in_sz = self.train_config.arch.trial_encoder.attention.out_sz
            else:
                self.train_config.arch.invertible.param_sz = len(self.fit_param)
                self.train_config.arch.invertible.block.cond_sz = self.train_config.arch.trial_encoder.attention.out_sz

        # Model name
        self.name = name if name is not None else \
            f"{'PTE' if self.train_config.point_estimation else 'DNE'}_{get_compact_timestamp_str()}"

        # Warmup
        self.point_estimation = self.train_config.point_estimation
        amortizer_fn = RegressionForTrialData if self.point_estimation else AmortizerForTrialData
        self.amortizer = amortizer_fn(config=self.train_config["arch"])

        self.lr = self.train_config["learning_rate"]
        self.lr_gamma = self.train_config["lr_gamma"]
        self.clipping = self.train_config["clipping"]
        self.optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(self.optimizer, T_0=10, T_mult=1, eta_max=self.lr, T_up=1, gamma=self.lr_gamma)

        self.model_path = os.path.join(DIR_TO_DATA, f"{self.name}/pts")
        self.board_path = os.path.join(DIR_TO_DATA, f"{self.name}/board")
        self.result_path = os.path.join(DIR_TO_DATA, f"{self.name}/results")

        if load_model is None:
            save_dict_to_yaml(self.train_config, os.path.join(DIR_TO_DATA, f"{self.name}/train_config.yaml"))
            save_dict_to_yaml(self.train_dataset.config['data'], os.path.join(DIR_TO_DATA, f"{self.name}/data_config.yaml"))
            save_dict_to_yaml(self.train_dataset.config['simul'], os.path.join(DIR_TO_DATA, f"{self.name}/simul_config.yaml"))

        self.clipping = float("Inf")

        if load_model is not None:
            self.load(load_model)
        
        self._log_history(f"[ Training Dataset Loaded ] Total parameter sets: {self.train_dataset.n_param}, trials per parameter set: {self.train_dataset.num_of_trial_per_param}\n")

    
    def process_data(self, data:np.ndarray):
        if self.train_config.data_grouping.enabled:
            args = (data, self.train_config.data_grouping.group_size, 
                    self.train_config.data_grouping.aggregation_features)
            if len(data.shape) == 2:
                return DriftDiffusionDataset.process_data(*args)
            elif len(data.shape) == 3:
                return DriftDiffusionDataset.process_batch_data(*args)
            else:
                raise ValueError(f"Data shape not supported for grouping: {data.shape}")
        else:
            return data
        

    def train(
        self,
        n_iter=50,
        step_per_iter=128,
        batch_sz=64,
        board=True,
        save_freq=10,
    ):
        """
        Training loop

        n_iter (int): Number of training iterations
        step_per_iter (int): Number of training steps per iteration
        batch_sz (int): Batch size
        board (bool): Whether to use tensorboard (default: True)
        """
        iter = self.iter
        last_step = self.iter * step_per_iter

        self.logger = Logger(
            self.name,
            last_step=last_step, 
            board=board, 
            board_path=self.board_path
        )

        # Training iterations
        losses = dict()
        print(f"\n[ Training - {self.name} ]")
        for iter in range(self.iter + 1, n_iter + 1):
            losses[iter] = []

            # Training loop
            with tqdm(total=step_per_iter, desc=f" Iter {iter}") as progress:
                for step in range(step_per_iter):
                    params, stats = self.train_dataset.sample(batch_sz=batch_sz)
                    stats = self.process_data(stats)
                    batch_args = (params, stats)

                    # Training step
                    loss = self._train_step(*batch_args)
                    losses[iter].append(loss)

                    # Logging
                    if step % 10 == 0:
                        self.logger.write_scalar(train_loss=loss, lr=self.scheduler.get_last_lr()[0])
                    progress.set_postfix_str(f"Avg.Loss: {np.mean(losses[iter]):.3f}")
                    progress.update(1)
                    self.logger.step()
                    self.scheduler.step((iter-1) + step/step_per_iter)
                    
                    if np.isnan(loss):
                        raise RuntimeError("Nan loss computed.")

            # Save model
            if iter % save_freq == 0:
                self.save(iter)
                valid_res, r_squared = self.valid()

                self.logger.write_scalar(**valid_res)
                self._log_history(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ")
                self._log_history(f"Iter {iter:03d}: Avg. Loss = {np.mean(loss):.6f} | ")
                self._log_history(f"Steps = {step_per_iter} | Batch Size = {batch_sz} | Recovery R2 = {np.mean(r_squared):.2f}\n")

            self.iter = iter

        print("\n[ Training Done ]")
        if iter in losses:
            print(f"  Training Loss: {np.mean(losses[iter])}\n")
        

    def _train_step(self, params, stat_data):
        self.amortizer.train()
        if self.point_estimation:
            params_tensor = torch.FloatTensor(params).to(self.amortizer.device)
            loss = F.mse_loss(self.amortizer(stat_data), params_tensor)
        else:
            z, log_det_J = self.amortizer(params, stat_data)
            loss = torch.mean(0.5 * torch.square(torch.norm(z, dim=-1)) - log_det_J)
        return self._optim_step(loss)
    

    def _optim_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.amortizer.parameters():
            param.grad.data.clamp_(-self.clipping, self.clipping)
        self.optimizer.step()
        return loss.item()
    

    def _log_history(self, log):
        log_path = os.path.join(DIR_TO_DATA, f"{self.name}/training_log.txt")
        with open(log_path, 'a') as f:
            f.write(log)


    def save(self, iter, path=None):
        """
        Save model, optimizer, and scheduler with iteration number
        """
        if path is None:
            os.makedirs(self.model_path, exist_ok=True)
            ckpt_path = f"{self.model_path}/iter{iter:03d}.pt"
        else:
            os.makedirs(path, exist_ok=True)
            ckpt_path = path + f"iter{iter:03d}.pt"
        torch.save({
            "iteration": iter,
            "model_state_dict": self.amortizer.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, ckpt_path)
        

    def load(self, model_name):
        """
        Load model, optimizer, and scheduler from the latest checkpoint
        """
        assert model_name is not None, "You must specify the model name to be loaded."
        import glob
        ckpt_paths = glob.glob(os.path.join(DIR_TO_DATA, f"{model_name}/pts/iter*.pt"))
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

        self.name = model_name
        self.model_path = os.path.join(DIR_TO_DATA, f"{self.name}/pts")
        self.board_path = os.path.join(DIR_TO_DATA, f"{self.name}/board")
        self.result_path = os.path.join(DIR_TO_DATA, f"{self.name}/results")

        ckpt = torch.load(ckpt_path, map_location=self.amortizer.device.type)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iter = ckpt["iteration"]
        self.scheduler.step(iter)

        print(f"[ amortizer - loaded checkpoint ] Model {model_name} - Iteration {iter}")
        self._log_history(f"\n[ Model Loaded: {model_name}, iteration {iter} ]\n")

        self.iter = iter
    

    def valid(
        self,
        n_sample=300,   # Sample for distribution estimation - necessary for DNE only
        verbose=True
    ):
        self.amortizer.eval()
        valid_res = dict()

        type_list = ["mode", "mean", "median"] if isinstance(self.amortizer, AmortizerForTrialData) \
            else ["mode"]
        
        for infer_type in type_list:
            # Parameter recovery from simulated

            sim_gt_params, sim_valid_data = self.val_dataset.sample()
            r_squared = self.parameter_recovery(
                valid_res,
                sim_gt_params,
                sim_valid_data,
                n_sample,
                infer_type,
                surfix="sim",
            )

        if verbose:
            print(f"- parameter recovery (simulated) - Avg. R2=({np.mean(r_squared):.2f})")

        return valid_res, r_squared


    def parameter_recovery(
        self,
        result_log,
        gt_params,
        valid_data,
        n_sample,
        infer_type,
        surfix="",
    ):
        basedir = f"{self.result_path}/iter{self.iter+1:03d}/recovery_{infer_type}/" \
            if isinstance(self.amortizer, AmortizerForTrialData) \
                else f"{self.result_path}/iter{self.iter+1:03d}/"
        os.makedirs(basedir, exist_ok=True)

        # Note: all parameters are normalized: -1 ~ 1
        n_param = gt_params.shape[0]
        inferred_params = list()

        for param_i in range(n_param):
            stat_i = self.process_data(valid_data[param_i])
            param_z = self.amortizer.infer(
                stat_i,
                n_sample=n_sample, 
                type=infer_type
            )
            param_z = self._clip_params(param_z)
            param_w = self.train_dataset._denormalize_param(param_z)
            inferred_params.append(param_w)
        inferred_params = np.array(inferred_params)

        gt_params = np.array([self.train_dataset._denormalize_param(p) for p in gt_params])
        
        npz_save(
            filename=os.path.join(basedir, f"inference_result.npz"),
            gt_params=gt_params,
            inferred_params=inferred_params
        )
        
        ax_r, ax_c = closest_factor_pair(len(self.fit_param))
        fig, axs = figure_grid(ax_r, ax_c, size_ax=3)
        axs = axs.ravel()
        rsq_list = list()

        for i, p in enumerate(self.fit_param):
            result = draw_r2_plot(
                ax=axs[i],
                xdata=gt_params[:,i],
                ydata=inferred_params[:,i],
                xlabel=f"True {SYMBOLS.get(p, p)}",
                ylabel=f"Inferred {SYMBOLS.get(p, p)}",
            )
            result_log[f"Parameter_Recovery/{infer_type}/r2_{p}_{surfix}"] = result["R2"]
            rsq_list.append(result["R2"])
        fig.suptitle(f"Num. of trials: {self.data_config.n_trial}", fontsize=12)

        figure_save(fig, f"{basedir}/recovery.png")

        return rsq_list
        

    def _clip_params(self, params):
        return np.clip(params, -1.0,  1.0)



class DriftDiffusionInferenceEngine:
    def __init__(self, model_name, ckpt=None, verbose=True):
        self.name = model_name
        self.root_path = os.path.join(DIR_TO_DATA, self.name)

        self.simul_config = load_config(os.path.join(self.root_path, "simul_config.yaml"))
        self.data_config = load_config(os.path.join(self.root_path, "data_config.yaml"))
        self.train_config = load_config(os.path.join(self.root_path, "train_config.yaml"))
        self.engine_config = self.train_config.arch

        # Inference engine setting
        if ckpt is None:
            model_list = glob(f"{self.root_path}/pts/iter*.pt")
            model_list.sort()
            model_path = model_list[-1]
            ckpt = int(os.path.basename(model_path)[4:-3])    # iter{n}.pt -> n
        else:
            model_path = f"{self.root_path}/pts/iter{ckpt:03d}.pt"

        amortizer_fn = RegressionForTrialData if self.train_config.point_estimation else AmortizerForTrialData

        self.engine = amortizer_fn(config=self.engine_config)
        ckpt = torch.load(model_path, map_location=self.engine.device.type)
        self.engine.load_state_dict(ckpt["model_state_dict"])
        self.engine.eval()
        iter = ckpt["iteration"]

        if verbose:
            print(f"[ inference engine - loaded model ] {model_name} - Iteration {iter}")

        # Simulator setting
        self.simulator = DriftDiffusionSimulator(config=self.simul_config)
        self.fit_param, self.fit_param_range = self.simulator.adjustable_parameters()

    
    def process_data(self, data:np.ndarray):
        data_norm = DriftDiffusionDataset._normalize_data_static(data, self.data_config)
        if self.train_config.data_grouping.enabled:
            args = (data_norm, self.train_config.data_grouping.group_size, 
                    self.train_config.data_grouping.aggregation_features)
            if len(data_norm.shape) == 2:
                return DriftDiffusionDataset.process_data(*args)
            elif len(data_norm.shape) == 3:
                return DriftDiffusionDataset.process_batch_data(*args)
            else:
                raise ValueError(f"Data shape not supported for grouping: {data_norm.shape}")
        else:
            return data_norm


    def infer(self, stat, n_sample=300, infer_type='mode', return_dict=True, return_samples=False):
        # Point estimation
        if self.train_config.point_estimation:
            assert not return_samples, "Point estimation model cannot return samples."
            z = self.engine.infer(self.process_data(stat), n_sample=n_sample, type=infer_type)
            post_sampled = None
        # Density estimation
        else:
            z, post_sampled = self.engine.infer(
                self.process_data(stat),
                n_sample=n_sample,
                type=infer_type,
                return_samples=True
            )

        z = np.clip(z, -1.0, 1.0)
        w = linear_denormalize(z, *np.array([self.fit_param_range[p] for p in self.fit_param]).T)

        if return_dict:
            z = dict(zip(self.fit_param, z))
            w = dict(zip(self.fit_param, w))
        
        return (z, w, post_sampled) if return_samples else (z, w)





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--simul_config', type=str, default='default', help='Simulator configuration')
    parser.add_argument('--data_config', type=str, default='default', help='Data configuration')
    parser.add_argument('--train_config', type=str, default='default', help='Training configuration')

    parser.add_argument('--gen_train_data', action='store_true', help='Generate training data')
    parser.add_argument('--gen_val_data', action='store_true', help='Generate validation data')
    parser.add_argument('--tr_n_param', type=int, default=8192, help='Number of parameter sets per training data file')
    parser.add_argument('--tr_n_file', type=int, default=1, help='Number of training data files to generate')
    parser.add_argument('--vd_n_param', type=int, default=100, help='Number of parameter sets for validation data')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs for data generation')

    parser.add_argument('--train', action='store_true', help='Train the amortized inference model')
    parser.add_argument('--name', type=str, default=None, help='Name of the model to be trained')
    parser.add_argument('--load_model', type=str, default=None, help='Load existing model for training')
    parser.add_argument('--n_iter', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--step_per_iter', type=int, default=128, help='Number of training steps per iteration')
    parser.add_argument('--batch_sz', type=int, default=64, help='Training batch size')
    parser.add_argument('--save_freq', type=int, default=25, help='Frequency of saving the model during training')

    args = parser.parse_args()

    if args.gen_train_data:
        dataset = DriftDiffusionTrainingDataset(
            simul_config=args.simul_config, 
            data_config=args.data_config, 
            loading_existing_data=False
        )
        dataset.generate_new_data(n_param_per_file=args.tr_n_param, repeat=args.tr_n_file, n_jobs=args.n_jobs)
    
    if args.gen_val_data:
        dataset = DriftDiffusionValidationDataset(
            simul_config=args.simul_config, 
            data_config=args.data_config, 
            loading_existing_data=False
        )
        dataset.generate_new_data(n_param=args.vd_n_param, n_jobs=args.n_jobs)

    if args.train:
        trainer = DriftDiffusionAmortizedInferenceTrainer(
            name=args.name, 
            load_model=args.load_model, 
            simul_config=args.simul_config, 
            data_config=args.data_config, 
            train_config=args.train_config
        )
        trainer.train(
            n_iter=args.n_iter,
            step_per_iter=args.step_per_iter, 
            batch_sz=args.batch_sz, 
            save_freq=args.save_freq
        )
    
