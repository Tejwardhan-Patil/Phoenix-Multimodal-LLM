import os
import pandas as pd
import json
from PIL import Image
import numpy as np
import librosa

def load_text_data(file_path):
    """
    Loads text data from a CSV or JSON file.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file format for text data. Use .csv or .json.")
    
    return preprocess_text_data(data)

def preprocess_text_data(data):
    """
    Preprocesses text data, tokenization and cleanup.
    """
    if isinstance(data, pd.DataFrame):
        data['text'] = data['text'].apply(tokenize_text)
    elif isinstance(data, dict):
        data = {k: tokenize_text(v) for k, v in data.items()}
    
    return data

def tokenize_text(text):
    """
    Tokenizes and preprocesses text.
    """
    tokens = text.lower().split()  # Simple whitespace tokenization
    return tokens

def load_image_data(image_dir, resize_shape=(224, 224)):
    """
    Loads and preprocesses images from a directory.
    """
    images = []
    metadata = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path)
            img = preprocess_image(img, resize_shape)
            img_array = np.array(img)
            images.append(img_array)
            metadata.append({'filename': img_file, 'size': img.size})
    return images, metadata

def preprocess_image(image, resize_shape):
    """
    Resizes and normalizes image.
    """
    image = image.resize(resize_shape)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return Image.fromarray((image_array * 255).astype(np.uint8))

def load_audio_data(audio_dir, audio_format='.wav', sample_rate=22050):
    """
    Loads and preprocesses audio data from a directory.
    """
    audio_data = []
    metadata = []
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith(audio_format):
            audio_path = os.path.join(audio_dir, audio_file)
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            audio_data.append(audio)
            metadata.append({'filename': audio_file, 'sample_rate': sr, 'duration': librosa.get_duration(audio, sr=sr)})
    return audio_data, metadata

def extract_audio_features(audio_data, sr):
    """
    Extracts audio features (MFCC, chroma, etc.).
    """
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return mfccs

def load_multimodal_data(text_path, image_dir, audio_dir, resize_shape=(224, 224), audio_format='.wav'):
    """
    Loads text, image, and audio data.
    """
    text_data = load_text_data(text_path)
    image_data, image_metadata = load_image_data(image_dir, resize_shape)
    audio_data, audio_metadata = load_audio_data(audio_dir, audio_format)
    
    log_dataset_statistics(text_data, image_data, audio_data)
    
    return {
        'text_data': text_data,
        'image_data': image_data,
        'image_metadata': image_metadata,
        'audio_data': audio_data,
        'audio_metadata': audio_metadata
    }

def log_dataset_statistics(text_data, image_data, audio_data):
    """
    Logs statistics about the dataset.
    """
    num_text_samples = len(text_data)
    num_images = len(image_data)
    num_audio_samples = len(audio_data)

    print(f"Text samples: {num_text_samples}")
    print(f"Image samples: {num_images}")
    print(f"Audio samples: {num_audio_samples}")

    if isinstance(text_data, pd.DataFrame):
        text_sample_lengths = text_data['text'].apply(len)
        print(f"Average text sample length: {text_sample_lengths.mean()} tokens")
    
    avg_image_size = np.mean([img.shape for img in image_data], axis=0)
    print(f"Average image size: {avg_image_size}")

    avg_audio_duration = np.mean([librosa.get_duration(y=audio) for audio in audio_data])
    print(f"Average audio duration: {avg_audio_duration} seconds")

def validate_file_existence(file_path):
    """
    Ensures file or directory exists.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

def load_data_from_multiple_sources(text_paths, image_dirs, audio_dirs):
    """
    Loads multimodal data from multiple directories/sources.
    """
    all_text_data = []
    all_image_data = []
    all_audio_data = []
    
    for text_path in text_paths:
        validate_file_existence(text_path)
        text_data = load_text_data(text_path)
        all_text_data.append(text_data)

    for image_dir in image_dirs:
        validate_file_existence(image_dir)
        images, _ = load_image_data(image_dir)
        all_image_data.extend(images)

    for audio_dir in audio_dirs:
        validate_file_existence(audio_dir)
        audio, _ = load_audio_data(audio_dir)
        all_audio_data.extend(audio)

    return all_text_data, all_image_data, all_audio_data

def save_preprocessed_data(output_dir, text_data=None, image_data=None, audio_data=None):
    """
    Saves preprocessed data to specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if text_data is not None:
        text_output_path = os.path.join(output_dir, 'text_data.csv')
        pd.DataFrame(text_data).to_csv(text_output_path, index=False)
    
    if image_data is not None:
        image_output_path = os.path.join(output_dir, 'image_data.npy')
        np.save(image_output_path, np.array(image_data))
    
    if audio_data is not None:
        audio_output_path = os.path.join(output_dir, 'audio_data.npy')
        np.save(audio_output_path, np.array(audio_data))

def load_preprocessed_data(input_dir):
    """
    Loads preprocessed data from specified directory.
    """
    text_data = None
    image_data = None
    audio_data = None

    text_path = os.path.join(input_dir, 'text_data.csv')
    if os.path.exists(text_path):
        text_data = pd.read_csv(text_path)
    
    image_path = os.path.join(input_dir, 'image_data.npy')
    if os.path.exists(image_path):
        image_data = np.load(image_path)

    audio_path = os.path.join(input_dir, 'audio_data.npy')
    if os.path.exists(audio_path):
        audio_data = np.load(audio_path)

    return text_data, image_data, audio_data

# Utility function to convert a list of tokens back into text
def tokens_to_text(tokens):
    return ' '.join(tokens)