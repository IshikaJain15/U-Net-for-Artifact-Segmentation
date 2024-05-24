import os
import random
from PIL import Image
import numpy as np
import os
import numpy as np
from PIL import Image

def extract_artifact_locations(mask_folder):
    """
    Extract artifact locations from the masks.
    
    Args:
    - mask_folder (str): Path to the folder containing label masks.
    
    Returns:
    - artifact_locations (list of np.ndarray): List of artifact locations for each mask.
    """
    artifact_locations = []
    for mask_filename in os.listdir(mask_folder):
        mask_path = os.path.join(mask_folder, mask_filename)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        # Get indices where mask value is greater than 0
        artifact_locations.append(np.where(mask_array > 0))
    return artifact_locations

def create_artifact_images(image_folder, mask_folder, output_folder, artifact_locations):
    """
    Create artifact images using the extracted artifact locations.
    
    Args:
    - image_folder (str): Path to the folder containing original images.
    - mask_folder (str): Path to the folder containing label masks.
    - output_folder (str): Path to the output folder to save the generated artifact images.
    - artifact_locations (list of np.ndarray): List of artifact locations for each mask.
    """
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over each image and its corresponding mask
    for image_filename, mask_locations in zip(os.listdir(image_folder), artifact_locations):
        image_path = os.path.join(image_folder, image_filename)
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Create an empty artifact image
        artifact_image = np.zeros_like(image_array)
        
        # Fill artifact locations with pixel values from the original image
        for x, y in zip(*mask_locations):
            artifact_image[x, y] = image_array[x, y]
        
        # Save the artifact image
        artifact_image_path = os.path.join(output_folder, f"artifact_{image_filename}")
        Image.fromarray(artifact_image).save(artifact_image_path)



def generate_artificial_image_with_artifacts(image_path, mask_path, artifact_mask_paths):
    """
    Generate an artificial image with random artifacts based on the provided image and mask.
    
    Args:
    - image_path (str): Path to the original image.
    - mask_path (str): Path to the label mask corresponding to the image.
    - artifact_mask_paths (list): List of paths to artifact masks.
    
    Returns:
    - artificial_image (PIL.Image.Image): Artificial image with random artifacts.
    """
    # Load original image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    # Convert mask to binary mask
    
    
    # Randomly select an artifact mask
    artifact_mask_path = random.choice(artifact_mask_paths)
    artifact_mask = Image.open(artifact_mask_path)
    binary_mask = np.array(artifact_mask) > 0
    # Extract artifact locations
    artifact_locations = np.where(np.array(artifact_mask) > 0)
    
    # Create a copy of the original image
    artificial_image = image.copy()
    
    # Add random artifacts to the artificial image
    for i in range(len(artifact_locations[0])):
        x, y = artifact_locations[0][i], artifact_locations[1][i]
        # Add the pixel value from the original image to the artifact location
        artificial_image.putpixel((x, y), image.getpixel((x, y)))
    
    return artificial_image

def generate_artificial_dataset(image_folder, mask_folder, artifact_mask_folder, output_folder, num_images):
    """
    Generate an artificial dataset with random artifacts.
    
    Args:
    - image_folder (str): Path to the folder containing original images.
    - mask_folder (str): Path to the folder containing label masks corresponding to the original images.
    - artifact_mask_folder (str): Path to the folder containing artifact masks.
    - output_folder (str): Path to the output folder to save the generated images.
    - num_images (int): Number of artificial images to generate.
    """
    # Get list of paths to original images and masks
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]
    mask_paths = [os.path.join(mask_folder, filename) for filename in os.listdir(mask_folder)]
    
    # Get list of paths to artifact masks
    artifact_mask_paths = [os.path.join(artifact_mask_folder, filename) for filename in os.listdir(artifact_mask_folder)]
    
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate artificial images
    for i in range(num_images):
        # Randomly select an original image and its corresponding mask
        image_path = random.choice(image_paths)
        mask_path = os.path.join(mask_folder, os.path.basename(image_path))
        
        # Generate artificial image with artifacts
        artificial_image = generate_artificial_image_with_artifacts(image_path, mask_path, artifact_mask_paths)
        
        # Save the generated image
        output_path = os.path.join(output_folder, f"artificial_image_{i}.png")
        artificial_image.save(output_path)
