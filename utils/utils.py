import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from augmentation import *
from utils.data_loading import *
# Function to display images and masks

def visualize(image, mask, transformed_image, transformed_mask):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Convert tensors to NumPy arrays
    '''image = image.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    transformed_image = transformed_image.detach().cpu().numpy()
    transformed_mask = transformed_mask.detach().cpu().numpy()'''

    axes[0, 0].imshow(image.squeeze(), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask.squeeze(), cmap='gray')
    axes[0, 1].set_title('Image after histogram equalization')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(transformed_image.squeeze(), cmap='gray')
    axes[1, 0].set_title('Image after adaptive histogram equalization')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(transformed_mask.squeeze(), cmap='gray')
    axes[1, 1].set_title('Mask')
    axes[1, 1].axis('off')

    plt.show()

def plot_img_and_mask(img, mask):
    classes =  1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()
