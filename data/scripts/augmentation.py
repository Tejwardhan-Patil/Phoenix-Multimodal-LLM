import random
import numpy as np
import cv2
import librosa
import os
from transformers import pipeline

# Augment text data using paraphrasing and synonym replacement
def augment_text(text, num_augmentations=5):
    paraphraser = pipeline("text2text-generation", model="t5-small")
    augmented_texts = []
    
    def replace_synonyms(text):
        # Implementation for synonym replacement
        words = text.split()
        synonyms = {"data": "information", "augmentation": "enhancement", "technique": "method"}
        replaced_text = " ".join([synonyms.get(word, word) for word in words])
        return replaced_text
    
    for _ in range(num_augmentations):
        # Use paraphrasing for augmentation
        paraphrased_text = paraphraser(text, max_length=100, do_sample=True)[0]['generated_text']
        # Perform synonym replacement
        synonym_text = replace_synonyms(paraphrased_text)
        augmented_texts.append(synonym_text)
    
    return augmented_texts

# Augment image data with additional transformations like brightness and noise
def augment_image(image_path, num_augmentations=5):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path {image_path} does not exist.")
    
    image = cv2.imread(image_path)
    augmented_images = []
    
    for _ in range(num_augmentations):
        rows, cols, _ = image.shape
        
        # Random rotation
        rotation_angle = random.uniform(-45, 45)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows))
        
        # Random horizontal and vertical flipping
        flip_type = random.choice([-1, 0, 1])  # -1: both, 0: vertical, 1: horizontal
        flipped_image = cv2.flip(rotated_image, flipCode=flip_type)
        
        # Random brightness adjustment
        brightness_factor = random.uniform(0.5, 1.5)
        bright_image = np.clip(flipped_image * brightness_factor, 0, 255).astype(np.uint8)
        
        # Adding Gaussian noise
        noise = np.random.normal(0, 25, bright_image.shape)
        noisy_image = np.clip(bright_image + noise, 0, 255).astype(np.uint8)
        
        augmented_images.append(noisy_image)
    
    return augmented_images

# Augment audio data using pitch shifting, time stretching, and white noise addition
def augment_audio(audio_path, sr=22050, num_augmentations=5):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio path {audio_path} does not exist.")
    
    audio, _ = librosa.load(audio_path, sr=sr)
    augmented_audios = []
    
    for _ in range(num_augmentations):
        # Random pitch shift
        pitch_shift = random.uniform(-5, 5)
        augmented_audio = librosa.effects.pitch_shift(audio, sr, n_steps=pitch_shift)
        
        # Random time stretching
        time_stretch = random.uniform(0.8, 1.5)
        augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=time_stretch)
        
        # Adding white noise
        noise_factor = 0.005 * np.random.randn(len(augmented_audio))
        noisy_audio = augmented_audio + noise_factor
        
        augmented_audios.append(noisy_audio)
    
    return augmented_audios

# Save augmented images to disk
def save_augmented_images(augmented_images, output_dir="data/augmented_images/"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, img in enumerate(augmented_images):
        output_path = os.path.join(output_dir, f"augmented_image_{i}.jpg")
        cv2.imwrite(output_path, img)

# Save augmented audios to disk
def save_augmented_audios(augmented_audios, sr, output_dir="data/augmented_audio/"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, audio in enumerate(augmented_audios):
        output_path = os.path.join(output_dir, f"augmented_audio_{i}.wav")
        librosa.output.write_wav(output_path, audio, sr)

# Function to augment entire text dataset
def augment_text_dataset(texts, num_augmentations=5):
    augmented_dataset = []
    for text in texts:
        augmented_texts = augment_text(text, num_augmentations)
        augmented_dataset.extend(augmented_texts)
    return augmented_dataset

# Function to augment entire image dataset
def augment_image_dataset(image_paths, num_augmentations=5):
    all_augmented_images = []
    for image_path in image_paths:
        augmented_images = augment_image(image_path, num_augmentations)
        all_augmented_images.extend(augmented_images)
    return all_augmented_images

# Function to augment entire audio dataset
def augment_audio_dataset(audio_paths, sr=22050, num_augmentations=5):
    all_augmented_audios = []
    for audio_path in audio_paths:
        augmented_audios = augment_audio(audio_path, sr, num_augmentations)
        all_augmented_audios.extend(augmented_audios)
    return all_augmented_audios

# Function to augment multimodal dataset
def augment_multimodal_dataset(texts, image_paths, audio_paths, num_augmentations=5, sr=22050):
    augmented_texts = augment_text_dataset(texts, num_augmentations)
    augmented_images = augment_image_dataset(image_paths, num_augmentations)
    augmented_audios = augment_audio_dataset(audio_paths, sr, num_augmentations)
    
    return {
        "texts": augmented_texts,
        "images": augmented_images,
        "audios": augmented_audios
    }

# Usage
if __name__ == "__main__":
    # Augment text data
    texts = [
        "Data augmentation is a technique used to artificially increase the size of a dataset.",
        "Image transformations can improve model robustness."
    ]
    augmented_texts = augment_text_dataset(texts)
    print("Augmented Texts:", augmented_texts)
    
    # Augment image data
    image_paths = ["data/images/sample_image_1.jpg", "data/images/sample_image_2.jpg"]
    augmented_images = augment_image_dataset(image_paths)
    save_augmented_images(augmented_images)
    
    # Augment audio data
    audio_paths = ["data/audio/sample_audio_1.wav", "data/audio/sample_audio_2.wav"]
    augmented_audios = augment_audio_dataset(audio_paths)
    save_augmented_audios(augmented_audios, sr=22050)
    
    # Augment multimodal dataset
    multimodal_data = augment_multimodal_dataset(texts, image_paths, audio_paths)
    print("Augmented Multimodal Data:", multimodal_data)