import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

from . import utils as latentode_utils
from .rnn_baselines import *
from .ode_rnn import *
from .create_latent_ode_model import create_LatentODE_model
from .ode_func import ODEFunc, ODEFunc_w_Poisson
from .diffeq_solver import DiffeqSolver
from .utils import compute_loss_all_batches
from .plotting import Visualizations

############################################
# Added for GluNet package
############################################
import optuna
import darts
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# utils for darts
from data_formatter.base import *
from utils.darts_dataset import *
from utils.darts_processing import *
from utils.darts_training import *
from utils.darts_evaluation import *
############################################

class ArgsParser():
	def __init__(self, **entries):
		"""
		Dummy class to convert dict of model parameters to object.
		Needed for compatibility with Latent ODE package.
	    """
		self.__dict__.update(entries)

class LatentODEWrapper():
    def __init__(self,
                 device: str = 'cpu',
                 latents: int = 10,
                 rec_dims: int = 20,
                 rec_layers: int = 1,
                 gen_layers: int = 3,
                 units: int = 100,
                 gru_units: int = 100) -> None:
        """
        Wrapper class for Latent ODE model.
        
        Parameters
        ----------
        device: str
            Device to use.
        latents: int
            Dimension of latent space.
        rec_dims: int
            Dimensionality of the recognition model (ODE or RNN).
        rec_layers: int
            Number of layers of the recognition model (ODE or RNN).
        gen_layers: int
            Number of layers of the generative model (ODE or RNN).
        units: int
            Number of units per layer in ODE func
        gru_units: int
            Number of units per layer in RNN
        """
	    # define params -- these are default for application to GluNet
        input_dim = 1
        classif_per_tp = False
        n_labels = 1
        args = {'latents': latents,
                'rec_dims': rec_dims,
                'rec_layers': rec_layers,
                'gen_layers': gen_layers,
                'units': units,
                'gru_units': gru_units,
                'z0_encoder': 'odernn',
                'extrap': True,
                'poisson': False,
                'classif': False,
                'linear_classif': False,
                'dataset': 'glunet'}
        args = ArgsParser(**args)
        # define model
        obsrv_std = 0.01
        obsrv_std = torch.Tensor([obsrv_std]).to(device)
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        self.model = create_LatentODE_model(args, 
                                            input_dim, 
                                            z0_prior, 
                                            obsrv_std, 
                                            device, 
                                            classif_per_tp = classif_per_tp,
                                            n_labels = n_labels,)
    
    def load(self, model_path: str,
             device: str) -> None:
        """
        Load model from path.
        
        Parameters
        ----------
        path: str
            Path to model.
        device: str
            Device to use.
        """
        latentode_utils.get_ckpt_model(model_path, self.model, device)   
        
    def fit(self, 
            train_dataset: SamplingDatasetDual,
            val_dataset: SamplingDatasetDual,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 100,
            num_samples: int = 100,
            device: str = 'cuda',
            model_path: str = None,
            trial: optuna.trial.Trial = None,
            logger: SummaryWriter = None,
            visualize: bool = False) -> None:
        """
        Fit the model to the data, using Optuna for hyperparameter tuning.
        
        Parameters
        ----------
        train_dataset: SamplingDatasetPast
            Training dataset.
        val_dataset: SamplingDatasetPast
            Validation dataset.
        learning_rate: float
            Learning rate for Adam.
        batch_size: int
            Batch size.
        epochs: int
            Number of epochs.
        num_samples: int
            Number of samples for infinite mixture
        device: str
            Device to use.
        model_path: str
            Path to save the model.
        trial: optuna.trial.Trial
            Trial for hyperparameter tuning.
        logger: SummaryWriter
            Tensorboard logger for logging.
        visualize: bool
            Whether to visualize the training.
        """

        viz = Visualizations(device) if visualize else None
        experimentID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
        num_batches = len(train_loader)
        num_batches_val = len(val_loader)
        train_loader = latentode_utils.inf_generator(train_loader)
        val_loader = latentode_utils.inf_generator(val_loader)

        optimizer = optim.Adamax(self.model.parameters(), lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler()
        best_loss = np.inf

        viz_batch = train_loader.__next__()
        inp_len, out_len = viz_batch[0].shape[1], viz_batch[-1].shape[1]
        mask = torch.ones(viz_batch[0].shape)
        mask[:, 20:31, ...] = 0
        print(torch.sum(mask == 0))
        mask = mask.to(device)
        observed_data = viz_batch[0].clone().to(device)
        observed_data[mask == 0] = 0
        observed_tp = torch.arange(0, inp_len).to(device) / 12
        viz_batch_dict = {
                            'observed_data': observed_data,
                            'observed_tp': observed_tp,
                            'data_to_predict': viz_batch[0].to(device),
                            'tp_to_predict': observed_tp,
                            'observed_mask': mask,
                            'mask_predicted_data': None,
                            'labels': None,
                            'mode': 'interp'
                        }
        for itr in range(1, num_batches * (epochs + 1)):
            optimizer.zero_grad()
            latentode_utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-3 / 10)

            wait_until_kl_inc = 10
            if itr // num_batches < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

            # convert to format expected by model
            with torch.cuda.amp.autocast():
                batch = train_loader.__next__()
                inp_len, out_len = batch[0].shape[1], batch[-1].shape[1]
                observed_data = batch[0].clone().to(device)
                observed_data[mask == 0] = 0
                observed_tp = torch.arange(0, inp_len).to(device) / 12
                batch_dict = {
                    'observed_data': observed_data,
                    'observed_tp': observed_tp,
                    'data_to_predict': batch[0].to(device),
                    'tp_to_predict': observed_tp,
                    'observed_mask': mask,
                    'mask_predicted_data': None,
                    'labels': None,
                    'mode': 'interp'
                }
                # compute loss
                train_res = self.model.compute_all_losses(batch_dict, n_traj_samples=3, kl_coef=kl_coef)
                loss = train_res["loss"]
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # log train loss
            if logger is not None:
                logger.add_scalar('train_loss', train_res["loss"].item(), itr)
                logger.add_scalar('train_likelihood', train_res["likelihood"].item(), itr)
                logger.add_scalar('train_mse', train_res["mse"].item(), itr)
            
            with torch.no_grad():
                # visualize
                if visualize and itr % 10 == 0:
                    viz.draw_all_plots_one_dim(viz_batch_dict, 
                                               self.model, 
                                               plot_name=f'iter_{itr}.png', 
                                               save=True,
                                               experimentID=experimentID)
                    plt.pause(0.1)

            