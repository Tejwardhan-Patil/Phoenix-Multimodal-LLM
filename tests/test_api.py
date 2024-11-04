import unittest
import requests
import time

class TestAPI(unittest.TestCase):

    base_url = "http://localhost:5000" 

    def test_text_input(self):
        """Test API handling of valid text input."""
        data = {
            "input_type": "text",
            "content": "This is a valid text input for the API."
        }
        response = requests.post(f"{self.base_url}/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text_output", response.json())

    def test_invalid_text_input(self):
        """Test API handling of invalid text input."""
        data = {
            "input_type": "text",
            "content": ""  # Invalid: empty text
        }
        response = requests.post(f"{self.base_url}/predict", json=data)
        self.assertEqual(response.status_code, 400)  # Expecting bad request for invalid input
        self.assertIn("error", response.json())

    def test_image_input(self):
        """Test API handling of image input."""
        with open("tests/test_image.jpg", "rb") as image_file:
            files = {'file': image_file}
            response = requests.post(f"{self.base_url}/predict", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn("image_output", response.json())

    def test_invalid_image_input(self):
        """Test API handling of invalid image input (non-image file)."""
        data = {
            "input_type": "image",
            "file": "not_an_image.txt"
        }
        response = requests.post(f"{self.base_url}/predict", json=data)
        self.assertEqual(response.status_code, 400)  # Expecting bad request
        self.assertIn("error", response.json())

    def test_audio_input(self):
        """Test API handling of audio input."""
        with open("tests/test_audio.wav", "rb") as audio_file:
            files = {'file': audio_file}
            response = requests.post(f"{self.base_url}/predict", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn("audio_output", response.json())

    def test_invalid_audio_input(self):
        """Test API handling of invalid audio input (corrupted file)."""
        with open("tests/invalid_audio.wav", "rb") as audio_file:
            files = {'file': audio_file}
            response = requests.post(f"{self.base_url}/predict", files=files)
        self.assertEqual(response.status_code, 400)  # Expecting bad request for corrupted file
        self.assertIn("error", response.json())

    def test_multimodal_input(self):
        """Test API handling of valid multimodal input (text, image, audio)."""
        files = {
            "input_type": "multimodal",
            "text_content": "A sentence for multimodal processing.",
            "image_file": open("tests/test_image.jpg", "rb"),
            "audio_file": open("tests/test_audio.wav", "rb")
        }
        response = requests.post(f"{self.base_url}/predict", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn("multimodal_output", response.json())

    def test_invalid_multimodal_input(self):
        """Test API handling of invalid multimodal input (missing files)."""
        data = {
            "input_type": "multimodal",
            "text_content": "Valid text but missing image and audio."
        }
        response = requests.post(f"{self.base_url}/predict", json=data)
        self.assertEqual(response.status_code, 400)  # Expecting bad request due to missing files
        self.assertIn("error", response.json())

    def test_text_performance(self):
        """Measure the response time for text input."""
        data = {
            "input_type": "text",
            "content": "A performance test for text input."
        }
        start_time = time.time()
        response = requests.post(f"{self.base_url}/predict", json=data)
        end_time = time.time()
        self.assertEqual(response.status_code, 200)
        self.assertIn("text_output", response.json())
        self.assertLess(end_time - start_time, 1, "Response took too long")

    def test_image_performance(self):
        """Measure the response time for image input."""
        with open("tests/test_image.jpg", "rb") as image_file:
            start_time = time.time()
            files = {'file': image_file}
            response = requests.post(f"{self.base_url}/predict", files=files)
            end_time = time.time()
        self.assertEqual(response.status_code, 200)
        self.assertIn("image_output", response.json())
        self.assertLess(end_time - start_time, 2, "Response took too long")

    def test_audio_performance(self):
        """Measure the response time for audio input."""
        with open("tests/test_audio.wav", "rb") as audio_file:
            start_time = time.time()
            files = {'file': audio_file}
            response = requests.post(f"{self.base_url}/predict", files=files)
            end_time = time.time()
        self.assertEqual(response.status_code, 200)
        self.assertIn("audio_output", response.json())
        self.assertLess(end_time - start_time, 2, "Response took too long")

    def test_multimodal_performance(self):
        """Measure the response time for multimodal input."""
        files = {
            "input_type": "multimodal",
            "text_content": "Performance test for multimodal input.",
            "image_file": open("tests/test_image.jpg", "rb"),
            "audio_file": open("tests/test_audio.wav", "rb")
        }
        start_time = time.time()
        response = requests.post(f"{self.base_url}/predict", files=files)
        end_time = time.time()
        self.assertEqual(response.status_code, 200)
        self.assertIn("multimodal_output", response.json())
        self.assertLess(end_time - start_time, 3, "Multimodal response took too long")

    def test_large_text_input(self):
        """Test handling of large text inputs."""
        large_text = "This is a very large text input. " * 500  # Generate a large input
        data = {
            "input_type": "text",
            "content": large_text
        }
        response = requests.post(f"{self.base_url}/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text_output", response.json())

    def test_large_image_input(self):
        """Test handling of large image files."""
        with open("tests/large_test_image.jpg", "rb") as large_image_file:
            files = {'file': large_image_file}
            response = requests.post(f"{self.base_url}/predict", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn("image_output", response.json())

    def test_large_audio_input(self):
        """Test handling of large audio files."""
        with open("tests/large_test_audio.wav", "rb") as large_audio_file:
            files = {'file': large_audio_file}
            response = requests.post(f"{self.base_url}/predict", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn("audio_output", response.json())

    def test_special_characters_in_text(self):
        """Test API handling of text with special characters."""
        data = {
            "input_type": "text",
            "content": "Text with special characters: @#$%^&*()!"
        }
        response = requests.post(f"{self.base_url}/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text_output", response.json())

    def test_no_input(self):
        """Test API handling of requests with no input data."""
        response = requests.post(f"{self.base_url}/predict", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_invalid_json_format(self):
        """Test API handling of invalid JSON format."""
        invalid_json = "{'input_type': 'text', 'content': 'missing double quotes'}"
        response = requests.post(f"{self.base_url}/predict", data=invalid_json)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_health_check(self):
        """Test API health check endpoint."""
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "ok")

    def test_latency_for_concurrent_requests(self):
        """Test API latency under concurrent requests."""
        start_time = time.time()
        for _ in range(10):
            data = {
                "input_type": "text",
                "content": "Concurrent request test"
            }
            response = requests.post(f"{self.base_url}/predict", json=data)
            self.assertEqual(response.status_code, 200)
            self.assertIn("text_output", response.json())
        end_time = time.time()
        self.assertLess(end_time - start_time, 10, "API latency is too high under concurrent requests")

if __name__ == "__main__":
    unittest.main()