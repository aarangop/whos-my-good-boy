
import os
import tensorflow as tf
from abc import ABC, abstractmethod


class ModelLoader(ABC):

    @abstractmethod
    def download(self, source: str):
        """Download the model from a specified path or source."""
        pass

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
        os.makedirs(models_dir, exist_ok=True)
        self.models_dir = models_dir
        self.models = {}

    def load(self, model_filename: str):
        """Load the model from a local path."""

        if model_filename in self.models:
            return self.models[model_filename]

        model_path = os.path.join(self.models_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file '{self.model_path}'")

        model = tf.keras.models.load_model(self.model_path)
        self.models[model_filename] = model

        return model

    def is_model_available(self):
        return self.model is not None and isinstance(self.model, tf.keras.models.Model)


class ModelService:
    def __init__(self):
        env = os.environ.get('ENVIRONMENT', 'development')

        if env == 'development':
            self.loader = LocalModelLoader(os.getenv("MODELS_DIR"))
        else:
            raise NotImplemented(f"Loader for {env} not implemented")

    def get_model(self, model_filename: str):
        self.loader.load(model_filename)
