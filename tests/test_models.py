import unittest
import torch
from models.text.gpt_model import GPTModel
from models.text.bert_model import BERTModel
from models.image.cnn_model import CNNModel
from models.image.vit_model import ViTModel
from models.audio.rnn_audio_model import RNNAudioModel
from models.audio.wavenet_model import WaveNetModel
from models.multimodal.text_image_model import TextImageModel
from models.multimodal.text_audio_model import TextAudioModel
from models.multimodal.fusion_model import FusionModel

class TestModels(unittest.TestCase):

    def setUp(self):
        # Set up common resources for all tests
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mock inputs for various modalities
        self.text_input = torch.randint(0, 1000, (4, 128)).to(self.device)  # Batch of 4 sequences, length 128
        self.image_input = torch.rand(4, 3, 224, 224).to(self.device)       # Batch of 4 images (3 channels, 224x224)
        self.audio_input = torch.rand(4, 1000).to(self.device)              # Batch of 4 audio sequences

        # Initialize models
        self.models = {
            'gpt': GPTModel().to(self.device),
            'bert': BERTModel().to(self.device),
            'cnn': CNNModel().to(self.device),
            'vit': ViTModel().to(self.device),
            'rnn_audio': RNNAudioModel().to(self.device),
            'wavenet': WaveNetModel().to(self.device),
            'text_image': TextImageModel().to(self.device),
            'text_audio': TextAudioModel().to(self.device),
            'fusion': FusionModel().to(self.device),
        }

    def test_gpt_model(self):
        output = self.models['gpt'](self.text_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], self.text_input.shape[0])
        self.assertEqual(output.shape[1], 128)  # Check the output sequence length

    def test_bert_model(self):
        output = self.models['bert'](self.text_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], self.text_input.shape[0])
        self.assertEqual(output.shape[1], 128)  # Ensure the output matches sequence length

    def test_cnn_model(self):
        output = self.models['cnn'](self.image_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], self.image_input.shape[0])
        self.assertGreaterEqual(output.shape[1], 10)  # Classification over 10+ classes

    def test_vit_model(self):
        output = self.models['vit'](self.image_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], self.image_input.shape[0])
        self.assertGreaterEqual(output.shape[1], 10)  # Check output for at least 10 classes

    def test_rnn_audio_model(self):
        output = self.models['rnn_audio'](self.audio_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], self.audio_input.shape[0])
        self.assertEqual(output.shape[1], 1000)  # Ensure audio output matches input sequence length

    def test_wavenet_model(self):
        output = self.models['wavenet'](self.audio_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], self.audio_input.shape[0])
        self.assertEqual(output.shape[1], 1000)  # Ensure the output matches input dimensions

    def test_text_image_model(self):
        output = self.models['text_image'](self.text_input, self.image_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], self.text_input.shape[0])
        self.assertEqual(output.shape[1], 128)  # Ensure output matches sequence length from text

    def test_text_audio_model(self):
        output = self.models['text_audio'](self.text_input, self.audio_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], self.text_input.shape[0])
        self.assertEqual(output.shape[1], 128)  # Ensure output matches sequence length from text

    def test_fusion_model(self):
        output = self.models['fusion'](self.text_input, self.image_input, self.audio_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], self.text_input.shape[0])
        self.assertEqual(output.shape[1], 128)  # Output should align with text input

    def test_text_input_edge_cases(self):
        # Test edge case: very short input text sequence
        short_text_input = torch.randint(0, 1000, (4, 10)).to(self.device)  # Sequence of length 10
        output = self.models['gpt'](short_text_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[1], 10)

        # Test edge case: very long input text sequence
        long_text_input = torch.randint(0, 1000, (4, 512)).to(self.device)  # Sequence of length 512
        output = self.models['bert'](long_text_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[1], 512)

    def test_image_input_edge_cases(self):
        # Test edge case: smaller images
        small_image_input = torch.rand(4, 3, 64, 64).to(self.device)  # Images of size 64x64
        output = self.models['cnn'](small_image_input)
        self.assertIsInstance(output, torch.Tensor)

        # Test edge case: larger images
        large_image_input = torch.rand(4, 3, 512, 512).to(self.device)  # Images of size 512x512
        output = self.models['vit'](large_image_input)
        self.assertIsInstance(output, torch.Tensor)

    def test_audio_input_edge_cases(self):
        # Test edge case: shorter audio sequences
        short_audio_input = torch.rand(4, 500).to(self.device)  # Audio sequences of length 500
        output = self.models['wavenet'](short_audio_input)
        self.assertIsInstance(output, torch.Tensor)

        # Test edge case: longer audio sequences
        long_audio_input = torch.rand(4, 2000).to(self.device)  # Audio sequences of length 2000
        output = self.models['rnn_audio'](long_audio_input)
        self.assertIsInstance(output, torch.Tensor)

    def test_multimodal_edge_cases(self):
        # Test with mismatched batch sizes for multimodal inputs
        mismatched_image_input = torch.rand(2, 3, 224, 224).to(self.device)  # Batch size 2 for image
        with self.assertRaises(ValueError):
            self.models['text_image'](self.text_input, mismatched_image_input)

        mismatched_audio_input = torch.rand(6, 1000).to(self.device)  # Batch size 6 for audio
        with self.assertRaises(ValueError):
            self.models['text_audio'](self.text_input, mismatched_audio_input)

if __name__ == '__main__':
    unittest.main()