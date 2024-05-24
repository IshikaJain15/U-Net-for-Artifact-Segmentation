from utils import *
import pandas as pd
import subprocess
import optuna
from sklearn.model_selection import train_test_split
import os
import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, MilanDataset
from utils.dice_score import dice_loss
from cross_val import cross_val_loop




# Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy 
def objective(trial):
    
    tuning_params = {
              'lr': trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3, 1e-2]),
              'epochs': trial.suggest_int("epochs", 50, 200, step=5),
              'batch_size': trial.suggest_int("batch_size", 4, 32, step=4)
              }
    
    _, cv_val_loss = cross_val_loop(train_dataset=train_dataset, test_dataset=test_dataset, tuning_params=tuning_params, n_trial=trial.number)
    print(cv_val_loss)
    return cv_val_loss

if __name__ == '__main__':
# This takes a lot of time to run, so do not try unless you have a GPU available
    dir_img = Path('./data/imgs/')
    dir_mask = Path('./data/masks/')

# 1. Create dataset
    try:
        train_dataset = MilanDataset(dir_img, dir_mask, scale=0.5)
    except (AssertionError, RuntimeError, IndexError):
        train_dataset = BasicDataset(dir_img, dir_mask, scale=0.5)
    dir_test_img = Path('./data/test/imgs/')
    dir_test_mask = Path('./data/test/masks/')
    test_dataset = BasicDataset(dir_test_img, dir_test_mask, scale=0.5)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=5)

    print(study.best_trial)

    df_trials = study.trials_dataframe().sort_values(by=['value'], ascending=True)
    print(df_trials)
    logging.info('saving the study')
    df_trials.to_csv(f'data/results/hyperparameter_tuning/hyperparameter_opt.csv', index=False)

    best_model, final_val_loss = cross_val_loop(train_dataset=train_dataset, test_dataset=test_dataset, tuning_params=study.best_params)
    print(best_model, final_val_loss)
    logging.info(best_model, final_val_loss)
    