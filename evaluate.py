import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import logging
from unet import UNet
import torchvision.transforms.functional as TF
from pathlib import Path
from utils.data_loading import BasicDataset, MilanDataset
from utils.dice_score import dice_loss
from utils.iou_score import *
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.utils import plot_img_and_mask, visualize
import os
from torch.utils.data import DataLoader
from hist_eq import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

def calculate_classification_metrics(y_true, y_pred):
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate accuracy, recall, precision, F1 score
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if tn + fp == 0:
        specificity = np.nan  # or any other appropriate value
    else:
        specificity = tn / (tn + fp)

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn)
    
    
    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score=0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            print(image.shape)
            logging.info(f'Image shape: {image.shape}')
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                mask_pred=torch.squeeze(mask_pred, dim=1)
                
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                iou_score+=pixel_wise_iou(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)

def test(net, dataloader, device, amp, i=1):
    net.eval()
    num_test_batches = len(dataloader)
    print(num_test_batches)
    dice_score = 0
    iou_score=0
    k=0.0
    ds=[]
    classification_metrics = {}
    true_labels=[]
    predicted_labels=[]
    # iterate over the test set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_test_batches, desc='Testing', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            
            # predict the mask
            mask_pred = net(image)
            mask_pred_binary = (F.sigmoid(mask_pred) > 0.5).float()
            mask_true_binary = (F.sigmoid(mask_true) > 0.5).float()
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                
                # compute the Dice score
                mask_pred = torch.squeeze(mask_pred, dim=1)
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                ds.append(dice_score.item())
                dice_scores=pd.DataFrame(ds)
                print(ds)
                iou_score+=pixel_wise_iou(mask_pred, mask_true, reduce_batch_first=False)
                k+=1
                print(dice_score/k)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
           
        
        # Check if any white pixel exists in the predicted mask
            predicted_label = (mask_pred_binary.sum(dim=(1, 2, 3)) > 0).int().cpu().numpy()  # Adjusted dimension for sum

        # True class labels based on the presence of an artifact in the true mask
            
            true_label=(mask_true_binary.sum(dim=(1, 2)) > 0).int().cpu().numpy()
            print(predicted_label, true_label)
            predicted_labels.append(predicted_label[0])
            true_labels.append(true_label[0])
            print(predicted_labels, true_labels)
        # Calculate classification metrics
            
        
        
        
    metrics = calculate_classification_metrics(true_labels, predicted_labels)
    metrics['dice score']= dice_score.item() / max(num_test_batches, 1)
    metrics['pixelwise iou score']= iou_score.item()/max(num_test_batches, 1)
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

# Save the results to an Excel file
    metrics_df.to_excel(f'data/results/re/class_metrics_train{i}.xlsx')
    print(metrics)
    print(ds)
    dice_scores=pd.DataFrame(ds)
    dice_scores.to_csv(f'data/results/re/test_dice_loss_train{i}.csv')
    print(iou_score/max(num_test_batches, 1))
    
    return dice_score / max(num_test_batches, 1), iou_score/max(num_test_batches, 1)

def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    dir_img = Path('./data/re/imgs/')
    dir_mask = Path('./data/re/masks/')
    dir_checkpoint = Path('./checkpoints/')
    dir_test_img = Path('./data/train_vgg/image')
    dir_test_mask = Path('./data/train_vgg/label2')
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    test_dataset = BasicDataset(dir_test_img, dir_test_mask, scale=0.5)
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    test_loader=DataLoader(test_dataset, shuffle=True, **loader_args)
    # Folder containing the model files
    folder_path = 'data/results/re'

# Find all model files in the folder
    model_files = glob.glob(os.path.join(folder_path, 'model*.pth'))
    i=0
# Iterate over the model files
    for model_file in model_files:

        
        i+=1
        model=model_file
        print(model)
        #model='best_he.pth'
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
        test_dice_score, test_iou_score=test(net, test_loader, device, amp=args.amp, i=i)
        logging.info(f'Test dice score={test_dice_score}')
        logging.info(f'Test iou score={test_iou_score}')
   
       
        