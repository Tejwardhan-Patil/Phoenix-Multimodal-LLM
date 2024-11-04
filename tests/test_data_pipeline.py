import unittest
import os
import numpy as np
from data.scripts.preprocess_text import preprocess_text
from data.scripts.preprocess_images import preprocess_image
from data.scripts.preprocess_audio import preprocess_audio
from data.processed.features import text_features, image_features, audio_features


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        # Set up any necessary paths or test files
        self.text_sample = "This is a test sentence for preprocessing."
        self.empty_text_sample = ""
        self.invalid_text_sample = None
        self.image_sample_path = os.path.join('data/raw/images/sample_image.jpg')
        self.invalid_image_path = os.path.join('data/raw/images/non_existent_image.jpg')
        self.audio_sample_path = os.path.join('data/raw/audio/sample_audio.wav')
        self.invalid_audio_path = os.path.join('data/raw/audio/non_existent_audio.wav')

    # ---- TEXT PREPROCESSING TESTS ----
    
    def test_preprocess_text(self):
        # Test valid text preprocessing
        processed_text = preprocess_text(self.text_sample)
        self.assertIsInstance(processed_text, list)
        self.assertGreater(len(processed_text), 0)
        self.assertTrue(all(isinstance(token, str) for token in processed_text))
        print(f"Processed text: {processed_text}")

    def test_empty_text_preprocessing(self):
        # Test handling of empty string in text preprocessing
        processed_text = preprocess_text(self.empty_text_sample)
        self.assertEqual(processed_text, [], "Empty text should return an empty list.")
        print(f"Empty text: {processed_text}")

    def test_invalid_text_preprocessing(self):
        # Test handling of invalid text input (None)
        with self.assertRaises(ValueError):
            preprocess_text(self.invalid_text_sample)
        print("Invalid text raised expected ValueError.")

    # ---- IMAGE PREPROCESSING TESTS ----
    
    def test_preprocess_image(self):
        # Test valid image preprocessing
        processed_image = preprocess_image(self.image_sample_path)
        self.assertIsInstance(processed_image, np.ndarray)
        self.assertEqual(processed_image.shape, (224, 224, 3))
        print(f"Processed image shape: {processed_image.shape}")

    def test_invalid_image_path(self):
        # Test handling of non-existent image file
        with self.assertRaises(FileNotFoundError):
            preprocess_image(self.invalid_image_path)
        print("Invalid image path raised expected FileNotFoundError.")

    def test_invalid_image_format(self):
        # Test handling of invalid image format (non-image file)
        invalid_format_image_path = os.path.join('data/raw/images/invalid_image.txt')
        with self.assertRaises(ValueError):
            preprocess_image(invalid_format_image_path)
        print("Invalid image format raised expected ValueError.")

    # ---- AUDIO PREPROCESSING TESTS ----
    
    def test_preprocess_audio(self):
        # Test valid audio preprocessing
        processed_audio = preprocess_audio(self.audio_sample_path)
        self.assertIsInstance(processed_audio, np.ndarray)
        self.assertEqual(processed_audio.shape[1], 128)
        print(f"Processed audio shape: {processed_audio.shape}")

    def test_invalid_audio_path(self):
        # Test handling of non-existent audio file
        with self.assertRaises(FileNotFoundError):
            preprocess_audio(self.invalid_audio_path)
        print("Invalid audio path raised expected FileNotFoundError.")

    def test_invalid_audio_format(self):
        # Test handling of invalid audio format (non-audio file)
        invalid_format_audio_path = os.path.join('data/raw/audio/invalid_audio.txt')
        with self.assertRaises(ValueError):
            preprocess_audio(invalid_format_audio_path)
        print("Invalid audio format raised expected ValueError.")

    # ---- FEATURE EXTRACTION TESTS ----
    
    def test_text_feature_extraction(self):
        # Test feature extraction for valid text
        features = text_features(self.text_sample)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 768)
        print(f"Text features shape: {features.shape}")

    def test_empty_text_feature_extraction(self):
        # Test feature extraction for empty text
        with self.assertRaises(ValueError):
            text_features(self.empty_text_sample)
        print("Empty text feature extraction raised expected ValueError.")

    def test_image_feature_extraction(self):
        # Test feature extraction for valid image
        processed_image = preprocess_image(self.image_sample_path)
        features = image_features(processed_image)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 2048)
        print(f"Image features shape: {features.shape}")

    def test_invalid_image_feature_extraction(self):
        # Test feature extraction for invalid image input
        with self.assertRaises(ValueError):
            image_features(None)
        print("Invalid image feature extraction raised expected ValueError.")

    def test_audio_feature_extraction(self):
        # Test feature extraction for valid audio
        processed_audio = preprocess_audio(self.audio_sample_path)
        features = audio_features(processed_audio)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 1024)
        print(f"Audio features shape: {features.shape}")

    def test_invalid_audio_feature_extraction(self):
        # Test feature extraction for invalid audio input
        with self.assertRaises(ValueError):
            audio_features(None)
        print("Invalid audio feature extraction raised expected ValueError.")

    # ---- MORE EDGE CASE TESTS ----
    
    def test_large_text_input(self):
        # Test handling of an extremely large text input
        large_text_sample = "word " * 10000  # A text with 10,000 repeated words
        processed_text = preprocess_text(large_text_sample)
        self.assertIsInstance(processed_text, list)
        self.assertEqual(len(processed_text), 10000)
        print(f"Processed large text length: {len(processed_text)}")

    def test_large_image_input(self):
        # Test handling of a very large image file
        large_image_path = os.path.join('data/raw/images/large_image.jpg')
        processed_image = preprocess_image(large_image_path)
        self.assertIsInstance(processed_image, np.ndarray)
        self.assertEqual(processed_image.shape, (224, 224, 3))
        print(f"Processed large image shape: {processed_image.shape}")

    def test_large_audio_input(self):
        # Test handling of a very large audio file
        large_audio_path = os.path.join('data/raw/audio/large_audio.wav')
        processed_audio = preprocess_audio(large_audio_path)
        self.assertIsInstance(processed_audio, np.ndarray)
        self.assertEqual(processed_audio.shape[1], 128)
        print(f"Processed large audio shape: {processed_audio.shape}")


if __name__ == '__main__':
    unittest.main()