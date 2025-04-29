import random
from typing import Dict
import os

import numpy as np

from app.services.base import BaseClassifierService
from app.core.errors import ModelNotLoadedError
from app.utils.inference_models.mobilenet_preprocessor import MobileNetProcessor
from app.utils.inference_models.model_loader_manager import ModelLoaderManager


class GeneralClassifierService(BaseClassifierService):
    """Service for general image classification"""

    def __init__(self):
        super().__init__()

        self.model_filename = os.getenv("CAT_DOG_OTHER_CLASSIFIER")
        self.model_loader = ModelLoaderManager.get_loader()
        self.model = self.model_loader.load(self.model_filename)
        self.model_processor = MobileNetProcessor()

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
        if self.model is None:
            raise ModelNotLoadedError("General classifier model not loaded")

        # Preprocess the image
        preprocessed_image = self.model_processor.preprocess_input(image_data)

        pred = self.model.predict(preprocessed_image)
        # For the placeholder implementation, return random classification results
        # This will be replaced with actual model inference when we add TensorFlow
        classes = ['cat', 'dog', 'other']
        decoded_preds = {class_name: pred[0][i]
                         for i, class_name in enumerate(classes)}
        # Normalize to ensure probabilities sum to 1
        total = np.sum(pred, axis=1)
        assert total == 1

        return decoded_preds
