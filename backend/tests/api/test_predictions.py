import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Sample test image path - we'll use this to load a test image
TEST_IMAGE_PATH = os.path.join(os.path.dirname(
    __file__), "../test_data/test_dog.jpg")

# Create test image directory if it doesn't exist
os.makedirs(os.path.dirname(TEST_IMAGE_PATH), exist_ok=True)


@pytest.fixture
def mock_image():
    """
    Create a mock image for testing or use an existing one if available
    """
    # If test image doesn't exist, create a blank one
    if not os.path.exists(TEST_IMAGE_PATH):
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='white')
        img.save(TEST_IMAGE_PATH)

    # Return the file path
    return TEST_IMAGE_PATH


class TestPredictionEndpoints:
    """Test cases for the prediction endpoints"""

    def test_classify_endpoint(self, client: TestClient, mock_image):
        """Test the general classification endpoint"""
        with open(mock_image, "rb") as f:
            response = client.post(
                "/api/v1/classify",
                files={"image": ("test_image.jpg", f, "image/jpeg")}
            )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "top_prediction" in data
        assert "processing_time" in data
        assert len(data["predictions"]) > 0

    def test_is_dog_endpoint(self, client: TestClient, mock_image):
        """Test the dog classification endpoint"""
        with open(mock_image, "rb") as f:
            response = client.post(
                "/api/v1/is-dog",
                files={"image": ("test_image.jpg", f, "image/jpeg")}
            )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "processing_time" in data
        assert data["prediction"] in ["dog", "not_dog"]
        assert 0 <= data["confidence"] <= 1

    def test_is_apolo_endpoint(self, client: TestClient, mock_image):
        """Test the Apolo classification endpoint"""
        with open(mock_image, "rb") as f:
            response = client.post(
                "/api/v1/is-apolo",
                files={"image": ("test_image.jpg", f, "image/jpeg")}
            )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "processing_time" in data
        assert data["prediction"] in ["apolo", "not_apolo"]
        assert 0 <= data["confidence"] <= 1

    def test_invalid_image_format(self, client: TestClient):
        """Test submitting an invalid file format"""
        # Create a text file instead of an image
        with open("test_text.txt", "w") as f:
            f.write("This is not an image")

        with open("test_text.txt", "rb") as f:
            response = client.post(
                "/api/v1/classify",
                files={"image": ("test_text.txt", f, "text/plain")}
            )

        # Clean up
        os.remove("test_text.txt")

        # Verify response
        assert response.status_code == 400

    def test_error_handling(self, client: TestClient, mock_image):
        """Test error handling when model inference fails"""
        # Mock the service to raise an exception
        with patch('app.services.general_classifier.GeneralClassifierService.predict',
                   side_effect=Exception("Test exception")):
            with open(mock_image, "rb") as f:
                response = client.post(
                    "/api/v1/classify",
                    files={"image": ("test_image.jpg", f, "image/jpeg")}
                )

            assert response.status_code == 500
