import re
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import librosa
import random

# Text Preprocessing Utilities
def clean_text(text):
    """
    Cleans the input text by removing unwanted characters, stopwords, and other noise.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces
    return text

def tokenize_text(text):
    """
    Tokenizes the input text into words or subwords.
    """
    return text.split()

def remove_stopwords(tokens, stopwords):
    """
    Removes stopwords from the tokenized text.
    """
    return [word for word in tokens if word not in stopwords]

def stem_words(tokens, stemmer):
    """
    Applies stemming to each token in the tokenized text.
    """
    return [stemmer.stem(token) for token in tokens]

def lemmatize_words(tokens, lemmatizer):
    """
    Applies lemmatization to each token in the tokenized text.
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

def ngram_generator(tokens, n):
    """
    Generates n-grams from tokenized text.
    """
    return zip(*[tokens[i:] for i in range(n)])

# Image Preprocessing Utilities
def normalize_image(image_path, size=(224, 224)):
    """
    Resizes and normalizes the image for model input.
    """
    img = Image.open(image_path)
    img = img.resize(size)
    img_array = np.array(img).astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    return img_array

def standardize_image(image_array):
    """
    Standardizes image pixels by subtracting the mean and dividing by standard deviation.
    """
    mean = np.mean(image_array, axis=(0, 1, 2))
    std = np.std(image_array, axis=(0, 1, 2))
    return (image_array - mean) / std

def augment_image(image_path):
    """
    Performs data augmentation by randomly flipping and rotating the image.
    """
    img = Image.open(image_path)
    
    # Random flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Random rotation
    angle = random.randint(-30, 30)
    img = img.rotate(angle)

    return img

def enhance_image_contrast(image_path, factor=1.5):
    """
    Enhances the contrast of the image.
    """
    img = Image.open(image_path)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
    return img

def random_crop_image(image, crop_size=(180, 180)):
    """
    Randomly crops the image to a specified size.
    """
    width, height = image.size
    left = random.randint(0, width - crop_size[0])
    top = random.randint(0, height - crop_size[1])
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    return image.crop((left, top, right, bottom))

# Audio Preprocessing Utilities
def load_audio(audio_path, sample_rate=22050):
    """
    Loads an audio file, resamples it to the target sample rate.
    """
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    return audio, sr

def extract_mfcc(audio, sample_rate, n_mfcc=13):
    """
    Extracts MFCC features from the audio signal.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc

def segment_audio(audio, segment_length=2.0, sample_rate=22050):
    """
    Segments audio into smaller chunks of fixed duration.
    """
    segment_size = int(segment_length * sample_rate)
    return [audio[i:i+segment_size] for i in range(0, len(audio), segment_size)]

def noise_reduction(audio, sample_rate):
    """
    Reduces noise from the audio signal using spectral gating.
    """
    noise_profile = np.mean(librosa.feature.melspectrogram(audio, sr=sample_rate), axis=1)
    reduced_audio = librosa.effects.remix(audio, intervals=librosa.effects.split(audio, top_db=30))
    return reduced_audio

def pitch_shift(audio, sample_rate, n_steps=2):
    """
    Shifts the pitch of the audio signal.
    """
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)

def time_stretch(audio, rate=1.25):
    """
    Stretches the time of the audio without changing pitch.
    """
    return librosa.effects.time_stretch(audio, rate)

def extract_chroma_features(audio, sample_rate):
    """
    Extracts chroma features from the audio signal.
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    return chroma

# Text Augmentation
def synonym_replacement(tokens, synonyms_dict, p=0.1):
    """
    Replaces words in the token list with their synonyms.
    """
    new_tokens = []
    for token in tokens:
        if token in synonyms_dict and random.random() < p:
            new_tokens.append(random.choice(synonyms_dict[token]))
        else:
            new_tokens.append(token)
    return new_tokens

def random_deletion(tokens, p=0.1):
    """
    Randomly deletes tokens from the token list.
    """
    if len(tokens) == 1:
        return tokens
    return [token for token in tokens if random.random() > p]

def random_insertion(tokens, insert_words, p=0.1):
    """
    Randomly inserts words into the token list.
    """
    new_tokens = tokens.copy()
    for _ in range(int(p * len(tokens))):
        new_tokens.insert(random.randint(0, len(new_tokens)), random.choice(insert_words))
    return new_tokens

# Advanced Image Processing
def convert_to_grayscale(image_path):
    """
    Converts an image to grayscale.
    """
    img = Image.open(image_path)
    grayscale_img = img.convert('L')
    return grayscale_img

def resize_image(image, target_size=(128, 128)):
    """
    Resizes the image to the target size.
    """
    return image.resize(target_size)

def apply_gaussian_blur(image_path, radius=2):
    """
    Applies Gaussian blur to an image.
    """
    img = Image.open(image_path)
    return img.filter(ImageFilter.GaussianBlur(radius))

# Text Cleaning Enhancements
def remove_special_characters(text, remove_digits=False):
    """
    Removes special characters from the text, with an option to remove digits.
    """
    pattern = r"[^a-zA-Z\s]" if not remove_digits else r"[^a-zA-Z0-9\s]"
    return re.sub(pattern, "", text)

def replace_numbers_with_words(text):
    """
    Replaces digits in the text with their corresponding words.
    """
    num_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
    }
    return re.sub(r'\d', lambda x: num_words[x.group()], text)

def correct_spelling(tokens, spell_checker):
    """
    Corrects spelling errors in the tokenized text.
    """
    return [spell_checker.correction(token) for token in tokens]

def shuffle_words(tokens):
    """
    Randomly shuffles the words in the tokenized text.
    """
    random.shuffle(tokens)
    return tokens

# Audio Data Augmentation
def add_background_noise(audio, noise_factor=0.005):
    """
    Adds random background noise to the audio signal.
    """
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def adjust_volume(audio, factor=1.5):
    """
    Adjusts the volume of the audio signal.
    """
    return audio * factor

def random_audio_crop(audio, crop_length, sample_rate):
    """
    Randomly crops a segment of the audio.
    """
    crop_samples = int(crop_length * sample_rate)
    start = random.randint(0, len(audio) - crop_samples)
    return audio[start:start + crop_samples]

# Summary of Utility Functions
def summarize_text(text, model):
    """
    Summarizes the input text using a pre-trained model.
    """
    return model.summarize(text)