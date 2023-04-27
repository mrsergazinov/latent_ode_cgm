import os
import sys
import argparse
import datetime
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import optuna
import darts
import random
from torch.utils.tensorboard import SummaryWriter

# load model
from latent_ode.trainer_glunet_interpol import LatentODEWrapper
from latent_ode.eval_glunet import test

# utils for darts
from utils.darts_training import print_callback
from utils.darts_dataset import SamplingDatasetDual, SamplingDatasetInferenceDual
from utils.darts_processing import load_data, reshuffle_data


import numpy as np
import matplotlib.pyplot as plt

def generate_random_samples(n, p, T, curve_params):
    '''
    Generate random samples from three distinct periodic curves with noise.

    Args:
    n: int, number of samples
    p: int, number of observations in the interval [0, T]
    T: float, end of the time interval
    curve_params: list of 3 dictionaries, each containing the keys 'amplitude', 'frequency', 'noise_std_dev'
                  e.g. [{'amplitude': 1, 'frequency': 1, 'noise_std_dev': 0.1}, ...]

    Returns:
    samples: numpy array, shape (n, p), random samples from three distinct periodic curves with noise
    '''

    t = np.linspace(0, T, p)
    samples = np.zeros((n, p))

    for i in range(n):
        curve_index = i % 3
        amplitude = curve_params[curve_index]['amplitude']
        frequency = curve_params[curve_index]['frequency']
        noise_std_dev = curve_params[curve_index]['noise_std_dev']
        
        curve = amplitude * np.sin(2 * np.pi * frequency * t)
        noise = np.random.normal(0, noise_std_dev, p)
        samples[i, :] = curve + noise

    return samples


if __name__ == '__main__':
   # Create data
    n = 200
    p = 100
    T = 1
    curve_params = [
        {'amplitude': 1, 'frequency': 4, 'noise_std_dev': 0.1},
        {'amplitude': 1, 'frequency': 8, 'noise_std_dev': 0.2},
        {'amplitude': 1, 'frequency': 12, 'noise_std_dev': 0.3}
    ]
    samples = generate_random_samples(n, p, T, curve_params)
    samples = samples.astype(np.float32)
    
    # Convert to series
    series = []
    t = pd.date_range(start='2020-01-01', periods=p, freq='D')
    for i in range(n):
        series.append(darts.TimeSeries.from_times_and_values(t, samples[i, :]))
    series = [s.with_static_covariates(pd.DataFrame([0])) for s in series]

    # Create dataset
    out_len = 12
    in_len = 48
    max_samples_per_ts = 100
    dataset_train = SamplingDatasetDual(series,
                                        series,
                                        output_chunk_length=out_len,
                                        input_chunk_length=in_len,
                                        use_static_covariates=True,
                                        max_samples_per_ts=max_samples_per_ts,)
    
    # Create model
    model = LatentODEWrapper(device = 'cuda',
                            latents = 5,
                            rec_dims = 50,
                            rec_layers = 3,
                            gen_layers = 3,
                            units = 300,
                            gru_units = 100)
    
    # train model
    model_path = 'output/model_simulation.ckpt'
    writer = SummaryWriter('output/tensorboard/simulation')
    model.fit(dataset_train,
                dataset_train,
                learning_rate = 1e-3,
                batch_size = 32,
                epochs = 100,
                num_samples = 2,
                device = 'cuda',
                model_path = model_path,
                trial = None,
                logger = writer,
                visualize=True,)