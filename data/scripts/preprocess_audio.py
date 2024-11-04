import os
import librosa
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_audio(file_path, sr=22050):
    """Load an audio file, supporting multiple formats."""
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        logging.info(f"Loaded {file_path}")
        return audio, sample_rate
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None, None

def noise_reduction(audio, noise_factor=0.005):
    """Apply noise reduction by adding random noise."""
    try:
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio
    except Exception as e:
        logging.error(f"Error in noise reduction: {e}")
        return audio

def advanced_noise_reduction(audio, sr, n_fft=1024):
    """Apply spectral subtraction for noise reduction."""
    try:
        stft = librosa.stft(audio, n_fft=n_fft)
        magnitude, phase = librosa.magphase(stft)
        noise_est = np.mean(magnitude, axis=1)
        magnitude = np.maximum(magnitude - noise_est[:, np.newaxis], 0)
        audio_denoised = librosa.istft(magnitude * phase)
        return audio_denoised
    except Exception as e:
        logging.error(f"Error in advanced noise reduction: {e}")
        return audio

def time_stretch(audio, stretch_rate=1.0):
    """Apply time stretching to an audio signal."""
    try:
        return librosa.effects.time_stretch(audio, stretch_rate)
    except Exception as e:
        logging.error(f"Error in time stretching: {e}")
        return audio

def pitch_shift(audio, sr, pitch_steps=4):
    """Shift the pitch of an audio signal."""
    try:
        return librosa.effects.pitch_shift(audio, sr, n_steps=pitch_steps)
    except Exception as e:
        logging.error(f"Error in pitch shifting: {e}")
        return audio

def audio_to_spectrogram(audio, sample_rate, n_fft=2048, hop_length=512, n_mels=128):
    """Convert audio signal to mel-spectrogram."""
    try:
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        return log_spectrogram
    except Exception as e:
        logging.error(f"Error in spectrogram conversion: {e}")
        return None

def save_spectrogram(spectrogram, output_path):
    """Save spectrogram as a .npy file."""
    try:
        np.save(output_path, spectrogram)
        logging.info(f"Saved spectrogram to {output_path}")
    except Exception as e:
        logging.error(f"Error saving spectrogram: {e}")

def preprocess_audio(file_path, output_dir, config):
    """Process a single audio file."""
    try:
        # Load and preprocess audio
        audio, sample_rate = load_audio(file_path, sr=config['sample_rate'])
        if audio is None:
            return
        
        # Ensure audio duration
        if config['target_duration']:
            audio = ensure_audio_duration(audio, sample_rate, config['target_duration'])
        
        # Apply augmentations
        if config['noise_reduction']:
            audio = noise_reduction(audio, config['noise_factor'])
        if config['advanced_noise_reduction']:
            audio = advanced_noise_reduction(audio, sample_rate)
        if config['time_stretch']:
            audio = time_stretch(audio, config['stretch_rate'])
        if config['pitch_shift']:
            audio = pitch_shift(audio, sample_rate, config['pitch_steps'])

        # Convert to spectrogram
        spectrogram = audio_to_spectrogram(audio, sample_rate, n_fft=config['n_fft'], hop_length=config['hop_length'], n_mels=config['n_mels'])
        if spectrogram is not None:
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_spectrogram.npy")
            save_spectrogram(spectrogram, output_file)

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")

def ensure_audio_duration(audio, sr, target_duration):
    """Pad or trim audio to ensure consistent duration."""
    try:
        audio_duration = librosa.get_duration(y=audio, sr=sr)
        if audio_duration < target_duration:
            pad_length = int((target_duration - audio_duration) * sr)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        elif audio_duration > target_duration:
            audio = audio[:int(target_duration * sr)]
        return audio
    except Exception as e:
        logging.error(f"Error ensuring audio duration: {e}")
        return audio

def preprocess_audio_directory(input_dir, output_dir, config):
    """Process all audio files in a directory with parallel execution."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3'))]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(preprocess_audio, f, output_dir, config): f for f in audio_files}
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                future.result()
                logging.info(f"Completed processing {file_path}")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Configuration for preprocessing
    config = {
        'sample_rate': 22050,
        'noise_reduction': True,
        'noise_factor': 0.005,
        'advanced_noise_reduction': True,
        'time_stretch': True,
        'stretch_rate': 1.2,
        'pitch_shift': True,
        'pitch_steps': 4,
        'n_fft': 2048,
        'hop_length': 512,
        'n_mels': 128,
        'target_duration': 5.0  # in seconds
    }

    input_directory = 'data/raw/audio/'
    output_directory = 'data/processed/audio/'
    
    preprocess_audio_directory(input_directory, output_directory, config)