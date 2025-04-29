import os
from loguru import logger
import tensorflow as tf
from abc import ABC, abstractmethod


class ModelLoader(ABC):

    @abstractmethod
    def load(self, model_filename: str):
        """Load the model from a specified path or source."""
        pass

    @abstractmethod
    def is_model_available(self):
        """Check if the model is available for inference."""
        pass


class LocalModelLoader(ModelLoader):
    """
    Load ML models from local storage. Used for development on local machine. 
    """

    def __init__(self, models_dir: str = './models'):
        if not os.path.exists(models_dir):
            raise ValueError(
                f"Models directory '{models_dir}' does not exist.")

        self.models_dir = models_dir
        self.models = {}

        logger.info(
            f"Instantiating local model loader with models directory '{self.models_dir}'"
        )

    def load(self, model_filename: str):
        """Load the model from a local path."""
        logger.info(f"Loading model '{model_filename}' from {self.models_dir}")
        if model_filename in self.models:
            return self.models[model_filename]

        model_path = os.path.join(self.models_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file '{model_path}'")

        model = tf.keras.models.load_model(model_path)
        self.models[model_filename] = model

        return model

    def is_model_available(self):
        """Check if any models are loaded and available."""
        return len(self.models) > 0 and all(isinstance(model, tf.keras.models.Model) for model in self.models.values())
