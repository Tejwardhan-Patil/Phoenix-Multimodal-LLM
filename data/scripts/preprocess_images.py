import os
from PIL import Image
import numpy as np

# Directory paths for raw and processed images
raw_image_dir = 'data/raw/images/'
processed_image_dir = 'data/processed/images/'

# Image preprocessing configurations
TARGET_SIZE = (224, 224)  
NORMALIZE = True  
SAVE_FORMAT = 'png' 
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png'] 

def list_images_in_directory(directory):
    """
    List all image files in a directory.
    Args:
        directory (str): The directory to scan for image files.
    Returns:
        List[str]: List of image file paths.
    """
    images = []
    for filename in os.listdir(directory):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in SUPPORTED_FORMATS:
            images.append(os.path.join(directory, filename))
    return images

def create_directory_if_not_exists(directory):
    """
    Create a directory if it doesn't exist.
    Args:
        directory (str): Path of the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_image(image_path):
    """
    Load an image from the specified path.
    Args:
        image_path (str): Path to the image file.
    Returns:
        PIL.Image: Loaded image.
    """
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def resize_image(image, target_size):
    """
    Resize the image to the target size.
    Args:
        image (PIL.Image): The image to resize.
        target_size (tuple): Target dimensions.
    Returns:
        PIL.Image: Resized image.
    """
    return image.resize(target_size)

def normalize_image(image_array):
    """
    Normalize image array pixel values to the range [0, 1].
    Args:
        image_array (np.ndarray): Image array to normalize.
    Returns:
        np.ndarray: Normalized image array.
    """
    return image_array / 255.0

def preprocess_image(image_path, target_size=TARGET_SIZE, normalize=NORMALIZE):
    """
    Preprocesses a single image by resizing and normalizing (optional).
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size to resize the image.
        normalize (bool): Whether to normalize pixel values.
    Returns:
        np.ndarray: Preprocessed image array.
    """
    img = load_image(image_path)
    
    if img is None:
        return None
    
    img = resize_image(img, target_size)
    
    img_array = np.array(img)

    if normalize:
        img_array = normalize_image(img_array)

    return img_array

def preprocess_images_in_directory(input_dir, output_dir):
    """
    Preprocess all images in the input directory and save to the output directory.
    Args:
        input_dir (str): Directory containing raw images.
        output_dir (str): Directory to save preprocessed images.
    """
    create_directory_if_not_exists(output_dir)
    
    image_paths = list_images_in_directory(input_dir)

    for image_path in image_paths:
        processed_image = preprocess_image(image_path)
        
        if processed_image is not None:
            output_filename = os.path.basename(image_path).replace('.jpg', f'.{SAVE_FORMAT}')
            output_path = os.path.join(output_dir, output_filename)
            processed_image_uint8 = (processed_image * 255).astype(np.uint8)
            Image.fromarray(processed_image_uint8).save(output_path)

def log_preprocessing_start():
    """
    Logs the start of the preprocessing process.
    """
    print("Starting image preprocessing...")

def log_preprocessing_completion(num_images, output_dir):
    """
    Logs the completion of the preprocessing process.
    Args:
        num_images (int): Number of images processed.
        output_dir (str): Output directory where images are saved.
    """
    print(f"Preprocessing completed for {num_images} images.")
    print(f"Processed images are saved in {output_dir}")

def log_error(message):
    """
    Logs an error message.
    Args:
        message (str): Error message to log.
    """
    print(f"ERROR: {message}")

def preprocess_single_image_pipeline(image_path, output_dir):
    """
    Preprocess a single image and save to the output directory.
    Args:
        image_path (str): Path to the image.
        output_dir (str): Directory to save the processed image.
    """
    processed_image = preprocess_image(image_path)

    if processed_image is None:
        log_error(f"Failed to preprocess image: {image_path}")
        return

    output_filename = os.path.basename(image_path).replace('.jpg', f'.{SAVE_FORMAT}')
    output_path = os.path.join(output_dir, output_filename)

    processed_image_uint8 = (processed_image * 255).astype(np.uint8)
    
    try:
        Image.fromarray(processed_image_uint8).save(output_path)
    except Exception as e:
        log_error(f"Error saving processed image {output_filename}: {e}")

def preprocess_all_images_pipeline(input_dir, output_dir):
    """
    Preprocess all images in the input directory and save them.
    Args:
        input_dir (str): Directory with raw images.
        output_dir (str): Directory to save processed images.
    """
    log_preprocessing_start()

    create_directory_if_not_exists(output_dir)

    image_paths = list_images_in_directory(input_dir)

    if len(image_paths) == 0:
        log_error(f"No images found in directory: {input_dir}")
        return

    for image_path in image_paths:
        preprocess_single_image_pipeline(image_path, output_dir)

    log_preprocessing_completion(len(image_paths), output_dir)

def display_image_info(image_path):
    """
    Display basic info about an image (dimensions, format).
    Args:
        image_path (str): Path to the image file.
    """
    img = load_image(image_path)
    if img:
        print(f"Image {os.path.basename(image_path)} - Format: {img.format}, Size: {img.size}")

def validate_image_format(image_path):
    """
    Validates if an image file is in a supported format.
    Args:
        image_path (str): Path to the image file.
    Returns:
        bool: True if format is supported, False otherwise.
    """
    file_ext = os.path.splitext(image_path)[1].lower()
    return file_ext in SUPPORTED_FORMATS

def main():
    """
    Main function to execute the image preprocessing pipeline.
    """
    print("Initializing preprocessing pipeline...")

    # Define input and output directories
    input_dir = raw_image_dir
    output_dir = processed_image_dir

    preprocess_all_images_pipeline(input_dir, output_dir)

if __name__ == '__main__':
    main()