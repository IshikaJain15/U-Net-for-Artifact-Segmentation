import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
import os
import shutil
import tifffile
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import rasterio
from rasterio.plot import show
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

def preprocess_image(image, desired_height=256, desired_width=256):
    """
    Preprocesses an image for transfer learning with a U-Net model.
    
    Args:
    - image_path (str): Path to the input image file.
    - desired_height (int): Desired height of the resized image.
    - desired_width (int): Desired width of the resized image.
    
    Returns:
    - tensor (torch.Tensor): Preprocessed image tensor ready for input to the U-Net model.
    """
   

    
    # Resize the image
    tensor = TF.resize(image, (desired_height, desired_width))
    
    
    return tensor

import torchvision.transforms as transforms
import cv2
import torch

class UNetTransform:
    def __init__(self, desired_height=224, desired_width=224):
        self.desired_height = desired_height
        self.desired_width = desired_width

    def __call__(self, image):
        # Convert image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert image to tensor
        tensor = transforms.functional.to_tensor(image)

        # Resize the image
        tensor = transforms.functional.resize(tensor, (self.desired_height, self.desired_width))


        return tensor


class CustomDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = None
        self.images = list(data_dict.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.data_dict[img_path]
   
        if not os.path.exists(img_path):
            print(f"Image file does not exist: {img_path}")
        if not os.path.exists(label_path):
            print(f"Image file does not exist: {label_path}")
        
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        
        
    
        self.transform= transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        self.transform_label= transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(56),
                transforms.ToTensor(),
            ])
        if self.transform:
            image = self.transform(image)
        if self.transform_label:    
            label = self.transform_label(label)
        return image, label


class CustomDataset_VGG(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = None
        self.images = list(data_dict.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label=self.data_dict[img_path]
   
        if not os.path.exists(img_path):
            print(f"Image file does not exist: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        
        
        
    
        self.transform= transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
    
        if self.transform:
            image = self.transform(image)
        
        return image, label

import os

def create_image_label_dict(image_folder, label_folder):
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)
    
    # Sort the file lists to ensure correspondence between images and labels
    
    
    image_label_dict = {}
    
    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)
        
        if os.path.isfile(image_path) and os.path.isfile(label_path):
            image_label_dict[image_path] = label_path
            
    return image_label_dict


'''
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(os.path.join(data_dir, "images"))
        self.labels = {}  # Dictionary to store labels
        self.artifacts = {}  # Dictionary to store artifact annotations
        
        # Load labels and artifact annotations
        for image_file in self.image_files:
            # Assuming labels are stored in a text file with image file names and corresponding labels
            with open(os.path.join(data_dir, "labels.txt"), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    image_name, label = line.strip().split(',')
                    self.labels[image_name] = int(label)
            
            # Assuming artifact annotations are stored as images with the same file names as the original images
            artifact_path = os.path.join(data_dir, "artifacts", image_file)
            if os.path.exists(artifact_path):
                self.artifacts[image_file] = artifact_path

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.data_dir, "images", img_name))
        label = self.labels[img_name]
        
        # If there are artifacts, load the annotation image
        if img_name in self.artifacts:
            artifact_image = Image.open(self.artifacts[img_name])
        else:
            artifact_image = None

        if self.transform:
            image = self.transform(image)
            if artifact_image:
                artifact_image = self.transform(artifact_image)

        return image, label, artifact_image

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])'''
def link_label1_to_image(train_images_dir, train_labels_dir):
    labels={}
    artifacts={}
# Get the list of filenames in the images directory
    train_image_filenames = os.listdir(train_images_dir)

# Assuming label filenames correspond to image filenames with "_label" appended
    train_label_filenames = os.listdir(train_labels_dir)

# Create full paths for images and labels
    train_image_paths = [os.path.join(train_images_dir, filename) for filename in train_image_filenames]
    train_label_paths = [os.path.join(train_labels_dir, filename) for filename in train_label_filenames]

    for idx in range(len(train_label_paths)):
        labels[train_image_paths[idx]]=1
        artifacts[train_image_paths[idx]]=train_label_paths[idx]
    return labels, artifacts

def link_label0_to_image(train_images_dir, save_label_dir):
    if not os.path.exists(save_label_dir):
        os.makedirs(save_label_dir)
    labels={}
    artifacts={}
# Get the list of filenames in the images directory
    train_image_filenames = os.listdir(train_images_dir)

# Assuming label filenames correspond to image filenames with "_label" appended
    

# Create full paths for images and labels
    train_image_paths = [os.path.join(train_images_dir, filename) for filename in train_image_filenames]
    train_label_paths=[]

    for idx in range(len(train_image_paths)):
        image = Image.open(train_image_paths[idx])
        artifact_image = Image.new('RGB', image.size, (0, 0, 0))
        train_label_paths.append(os.path.join(save_label_dir, train_image_filenames[idx]))
        if not os.path.exists(train_label_paths[idx]):
            artifact_image.save(os.path.join(save_label_dir, train_image_filenames[idx]))  
        labels[train_image_paths[idx]]=0
        artifacts[train_image_paths[idx]]=train_label_paths[idx]
    return labels, artifacts

def split_image_into_squares(image, label_image, num_rows=3, num_cols=3):
    # Get the dimensions of each square
    square_height = image.shape[0] // num_rows
    square_width = image.shape[1] // num_cols
    
    # Initialize lists to store the cropped squares
    cropped_images = []
    cropped_label_images = []

    for r in range(num_rows):
        for c in range(num_cols):
            # Define the coordinates for cropping
            start_y = r * square_height
            end_y = (r + 1) * square_height
            start_x = c * square_width
            end_x = (c + 1) * square_width
            
            # Crop the original image
            cropped_image = image[start_y:end_y, start_x:end_x]
            cropped_images.append(cropped_image)
            
            # Crop the label image
            cropped_label = label_image[start_y:end_y, start_x:end_x]
            cropped_label_images.append(cropped_label)

    return cropped_images, cropped_label_images

def view_cropped_images(cropped_images, cropped_label_images):
    for i, cropped_image in enumerate(cropped_images):
        cv2.imshow(f"Cropped Image {i}", cropped_image)

    # Display the cropped label images
    for i, cropped_label in enumerate(cropped_label_images):
        cv2.imshow(f"Cropped Label {i}", cropped_label)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)    


def save_cropped_images_and_labels(image_paths_and_labels, output_folder):
    # Create output folders if they do not exist
    create_folder_if_not_exists(output_folder)
    create_folder_if_not_exists(os.path.join(output_folder, 'cropped_images'))
    create_folder_if_not_exists(os.path.join(output_folder, 'cropped_labels'))
    dict={}
    
    # Iterate over each image and label path in the dictionary
    for image_path, label_path in image_paths_and_labels.items():
        # Split the image into squares and get cropped images and labels
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        cropped_images, cropped_labels = split_image_into_squares(image, label)

        # Save cropped images and labels
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        for i, (cropped_image, cropped_label) in enumerate(zip(cropped_images, cropped_labels), 1):
            cropped_image_path = os.path.join(output_folder, 'cropped_images', f'{image_name}_cropped_{i}.png')
            cropped_label_path = os.path.join(output_folder, 'cropped_labels', f'{image_name}_cropped_{i}.png')
            if not os.path.exists(cropped_image_path):
                cv2.imwrite(cropped_image_path, cropped_image)

                # Check the threshold of the label image
                if cv2.countNonZero(cropped_label) > 200:
                    cv2.imwrite(cropped_label_path.replace('.png', '_label1.png'), cropped_label)
                    c_label_path=cropped_label_path.replace('.png', '_label1.png')
                
                else:
                    cv2.imwrite(cropped_label_path.replace('.png', '_label0.png'), cropped_label)
                    c_label_path=cropped_label_path.replace('.png', '_label0.png')
                dict[cropped_image_path]=c_label_path
            else:
                if not os.path.exists(cropped_label_path.replace('.png', '_label0.png')):
                    dict[cropped_image_path]=cropped_label_path.replace('.png', '_label1.png')
                else:
                    dict[cropped_image_path]=cropped_label_path.replace('.png', '_label0.png')
      
    return dict