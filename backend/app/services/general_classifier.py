import random
from typing import Dict

from app.services.base import BaseClassifierService
from app.core.errors import ModelNotLoadedError


class GeneralClassifierService(BaseClassifierService):
    """Service for general image classification"""

    def __init__(self):
        super().__init__()
        self.load_model()

    def predict(self, image_data: bytes) -> Dict[str, float]:
        """
        Classify an image using a pre-trained model

        Args:
            image_data: Raw bytes of the uploaded image

        Returns:
            Dictionary mapping class names to probabilities

        Raises:
            ModelNotLoadedError: If the model is not loaded
            InvalidImageError: If the image cannot be processed
        """
        if not self.model_loaded:
            raise ModelNotLoadedError("General classifier model not loaded")

        # Preprocess the image
        processed_image = self.preprocess_image(image_data)

        # For the placeholder implementation, return random classification results
        # This will be replaced with actual model inference when we add TensorFlow
        mock_classes = {
            "dog": random.uniform(0.1, 0.9),
            "cat": random.uniform(0.1, 0.5),
            "car": random.uniform(0.0, 0.3),
            "tree": random.uniform(0.0, 0.2),
            "person": random.uniform(0.0, 0.4)
        }

        # Normalize to ensure probabilities sum to 1
        total = sum(mock_classes.values())
        mock_classes = {k: v/total for k, v in mock_classes.items()}

        return mock_classes
