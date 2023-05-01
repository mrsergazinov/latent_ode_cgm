import os
import sys
import argparse
import datetime
from functools import partial

import numpy as np
import torch
import optuna
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter

# load model
from latent_ode.trainer_glunet_interpol import LatentODEWrapper
from latent_ode.eval_glunet import test

# utils for darts
from utils.darts_training import print_callback
from utils.darts_dataset import SamplingDatasetDual, SamplingDatasetInferenceDual
from utils.darts_processing import load_data, reshuffle_data


if __name__ == '__main__':
    # load data
    formatter, series, scalers = load_data(seed=0, 
                                            study_file=None, 
                                            dataset='hall',
                                            use_covs=True, 
                                            cov_type='dual',
                                            use_static_covs=True)

    # create datasets
    out_len = 12
    in_len = 48
    max_samples_per_ts = 100
    dataset_train = SamplingDatasetDual(series['train']['target'],
                                        series['train']['future'],
                                        output_chunk_length=out_len,
                                        input_chunk_length=in_len,
                                        use_static_covariates=True,
                                        max_samples_per_ts=max_samples_per_ts,)
    dataset_test = SamplingDatasetInferenceDual(target_series=series['test']['target'],
                                                covariates=series['test']['dynamic'],
                                                n=out_len,
                                                input_chunk_length=in_len,
                                                output_chunk_length=out_len,
                                                use_static_covariates=False,
                                                max_samples_per_ts = None,
                                                array_output_only=True,)
    
    # create model
    model = LatentODEWrapper(device = 'cuda',
                            latents = 5,
                            rec_dims = 50,
                            rec_layers = 3,
                            gen_layers = 3,
                            units = 300,
                            gru_units = 100)
    
    # train model
    model_path = 'output/model_glucose_interpol.ckpt'
    writer = SummaryWriter('output/tensorboard/glucose_interpol')
    # model.fit(dataset_train,
    #             dataset_train,
    #             learning_rate = 1e-3,
    #             batch_size = 32,
    #             epochs = 100,
    #             num_samples = 2,
    #             device = 'cuda',
    #             model_path = model_path,
    #             trial = None,
    #             logger = writer,
    #             visualize=True,)
    model.load(model_path, device='cuda')
    
    # evaluate model
    predictions = model.predict(dataset_test,
                                batch_size=32,
                                num_samples=10,
                                device='cuda')
    trues = np.array([dataset_test[i][0][20:31, :] for i in range(len(dataset_test))]) # interpolated chunk
    trues = (trues - scalers['target'].min_) / scalers['target'].scale_
    predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
    obsrv_std = 0.01 / scalers['target'].scale_
    id_errors_sample, id_likelihood_sample, id_cal_errors_sample = test(trues, predictions, obsrv_std)
    print(f'Mean of errors (MSE / MAE): {np.mean(id_errors_sample, axis=0)}')
    print(f'Median of errors (MSE / MAE): {np.median(id_errors_sample, axis=0)}')