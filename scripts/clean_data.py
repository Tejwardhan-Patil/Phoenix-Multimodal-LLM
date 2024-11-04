import os
import shutil
import logging
from pathlib import Path
import time
import hashlib

# Set up logging with timestamps for detailed tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories for raw and processed data
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Modality list for multimodal data
MODALITIES = ['text', 'images', 'audio']

# Hash map to track processed files to avoid duplication
processed_files = {}

def setup_environment():
    """Ensure the raw and processed directories are set up correctly."""
    logging.info("Setting up environment and directories...")
    for modality in MODALITIES:
        raw_modality_dir = RAW_DATA_DIR / modality
        processed_modality_dir = PROCESSED_DATA_DIR / modality

        if not raw_modality_dir.exists():
            raw_modality_dir.mkdir(parents=True)
            logging.info(f"Created directory: {raw_modality_dir}")

        if not processed_modality_dir.exists():
            processed_modality_dir.mkdir(parents=True)
            logging.info(f"Created directory: {processed_modality_dir}")

def get_file_hash(file_path):
    """Generate a hash for a file to track duplicates."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as file:
            buf = file.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        logging.error(f"Error generating hash for {file_path}: {e}")
        return None

def clean_modality_data(modality):
    """Clean and validate raw data for a specific modality, moving it to the processed folder."""
    raw_dir = RAW_DATA_DIR / modality
    processed_dir = PROCESSED_DATA_DIR / modality

    if not raw_dir.exists():
        logging.warning(f"Raw data directory for {modality} does not exist. Skipping.")
        return

    if not any(raw_dir.iterdir()):
        logging.warning(f"No raw data found for {modality}. Skipping.")
        return

    for file in raw_dir.iterdir():
        if file.is_file():
            logging.info(f"Processing file: {file}")
            file_hash = get_file_hash(file)
            if file_hash and file_hash in processed_files:
                logging.info(f"File {file} is a duplicate. Skipping.")
                continue

            try:
                clean_and_validate_file(file, modality)
                shutil.move(str(file), str(processed_dir))
                logging.info(f"Processed and moved file: {file}")
                processed_files[file_hash] = file
            except Exception as e:
                logging.error(f"Error processing {file}: {e}")

def clean_and_validate_file(file, modality):
    """Perform modality-specific cleaning and validation."""
    if modality == 'text':
        validate_text_file(file)
    elif modality == 'images':
        validate_image_file(file)
    elif modality == 'audio':
        validate_audio_file(file)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def validate_text_file(file):
    """Validate a text file by checking if it's non-empty and removing bad characters."""
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"File {file} is empty.")
            logging.info(f"Validated text file: {file}")
    except Exception as e:
        raise ValueError(f"Error validating text file {file}: {e}")

def validate_image_file(file):
    """Validate image files by checking their format and dimensions."""
    from PIL import Image

    valid_formats = ['.jpg', '.jpeg', '.png']
    if file.suffix.lower() not in valid_formats:
        raise ValueError(f"Invalid image format for file {file}. Accepted formats: {valid_formats}")

    try:
        with Image.open(file) as img:
            img.verify()
            logging.info(f"Validated image file: {file}")
    except Exception as e:
        raise ValueError(f"Error validating image file {file}: {e}")

def validate_audio_file(file):
    """Validate audio files by checking their format."""
    valid_formats = ['.wav', '.mp3']
    if file.suffix.lower() not in valid_formats:
        raise ValueError(f"Invalid audio format for file {file}. Accepted formats: {valid_formats}")

    try:
        logging.info(f"Validated audio file: {file}")
    except Exception as e:
        raise ValueError(f"Error validating audio file {file}: {e}")

def check_disk_space():
    """Check if there's enough disk space before processing data."""
    total, used, free = shutil.disk_usage("/")
    gb_free = free // (2**30)
    logging.info(f"Available disk space: {gb_free} GB")

    if gb_free < 10:
        raise OSError("Insufficient disk space. At least 10 GB required.")

def log_cleanup_summary():
    """Log a summary of the cleanup process after all modalities have been processed."""
    logging.info("Cleanup summary:")
    logging.info(f"Total files processed: {len(processed_files)}")
    logging.info("Process completed.")

def handle_failed_file(file, modality):
    """Move any failed files to a separate folder for manual review."""
    failed_dir = RAW_DATA_DIR / "failed" / modality
    failed_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(file), str(failed_dir))
    logging.error(f"Moved {file} to {failed_dir} for manual review.")

def clean_data():
    """Main function to clean and organize data across all modalities."""
    logging.info("Starting the data cleaning process...")

    try:
        check_disk_space()
        setup_environment()

        for modality in MODALITIES:
            logging.info(f"Cleaning data for modality: {modality}")
            try:
                clean_modality_data(modality)
            except Exception as e:
                logging.error(f"Error cleaning data for {modality}: {e}")
        log_cleanup_summary()

    except Exception as e:
        logging.critical(f"Data cleaning process failed: {e}")

    logging.info("Data cleaning process completed.")

if __name__ == "__main__":
    start_time = time.time()
    clean_data()
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")