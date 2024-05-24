import time
import os
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
import logging
from pathlib import Path
import glob
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
from evaluate import evaluate, test
from unet import UNet
from utils.data_loading import BasicDataset, MilanDataset
from utils.dice_score import dice_loss
from train import train_model

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()

def cross_val_loop(train_dataset, test_dataset, tuning_params, k=3, model_save_path='data/results/hyperparameter_tuning/best_cv_model.pth', n_trial=0):
    """
        Implements a cross validation loop for the image classifier.

        User can specify the scoring metric for the choosing of the
        best model between the cross entropy loss and the AUC.

        The parameter k is the number of folds to be implemented.

        The tuning_params argument requires a dictionary with the 
        keys as parameters and values as the respective parameter
        value to be used. The following tuning parameters are 
        required:
            - optimizer (name from torch.optim)
            - lr        (learning rate for the optimizer)
            - L2        (weight decay, L2 penalizer)
            - batch_size
            - num_epochs
    """
    kf = KFold(n_splits=k)
    i=n_trial
    cv_val_loss_values = []
    cv_val_auc_values = []
    
    best_score = None

    print('Running Trial with:')
    for key, value in tuning_params.items():
        print(f"{key}: {value}")
    args = get_args()
    

    # Get the length of the dataset
    dataset_length = len(train_dataset)

# Enumerate through each fold
    for fold, (train_index, val_index) in enumerate(kf.split(range(dataset_length))):
        print("\n\nStarting fold {}/{}\n".format(fold+1, k))
    
    # Create train and validation datasets for the fold
        train_fold = [{ 'image': train_dataset[i]['image'], 'mask': train_dataset[i]['mask'] } for i in train_index]
        val_fold = [{ 'image': train_dataset[i]['image'], 'mask': train_dataset[i]['mask'] } for i in val_index]

    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
        model = model.to(memory_format=torch.channels_last)
        model.to(device=device)
        loader_args = dict(batch_size=tuning_params['batch_size'], num_workers=os.cpu_count(), pin_memory=True)
        train_loader = DataLoader(train_fold, shuffle=True, **loader_args)
        val_loader = DataLoader(val_fold, shuffle=False, drop_last=True, **loader_args)
    

    
        train_model(model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer = optim.Adam(model.parameters(), lr=tuning_params['lr'], weight_decay= 1e-9, foreach=True),
            criterion=nn.BCEWithLogitsLoss(),
            epochs=tuning_params['epochs'],
            batch_size=tuning_params['batch_size'],
            learning_rate=tuning_params['lr'],
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            img_scale=args.scale,
            val_percent=args.val/ 100,
            n_trial=i)
        loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
        test_loader=DataLoader(test_dataset, shuffle=True, **loader_args)
        model=f'data/results/hyperparameter_tuning/model_{i}.pth'
        net = UNet(n_channels=1, n_classes=1, bilinear=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Loading model {model}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        state_dict = torch.load(model, map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)

        logging.info('Model loaded!')
        logging.info('Calculating test loss')
        test_dice_score, test_iou=test(net, test_loader, device, amp=args.amp, i=i)
        logging.info(f'Test dice score={test_dice_score}, Iou_score={test_iou}')

        cv_val_loss_values.append(test_dice_score.item())

        if best_score is None:
            best_model = model
            best_score = cv_val_loss_values[-1]

        
            
        else:
            if round(cv_val_loss_values[-1], 4) < round(best_score, 4):
                best_score = cv_val_loss_values[-1]
                best_model = model
    net = UNet(n_channels=1, n_classes=1, bilinear=False)     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(best_model, map_location=device)
    torch.save(net, model_save_path)
    mean_cv_val_loss = np.array(cv_val_loss_values).mean()
    

    print("Finished {k}-fold CV")
    print(f"Mean CV Loss: {mean_cv_val_loss}")
  

    return best_model, mean_cv_val_loss