
import os

from app.inference_models.model_loader import LocalModelLoader


class ModelLoaderManager:

    _loader = None

    @classmethod
    def get_loader(cls):

        env = os.environ.get('ENVIRONMENT', 'development')

        if cls._loader is not None:
            return cls._loader

        if env == 'development':
            cls._loader = LocalModelLoader(os.getenv("MODELS_DIR"))
        else:
            raise NotImplementedError(f"Loader for {env} not implemented")

        return cls._loader
