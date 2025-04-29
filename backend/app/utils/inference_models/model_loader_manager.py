import os
from app.core.config import settings
from app.utils.inference_models.model_loader import LocalModelLoader


class ModelLoaderManager:

    _loader = None

    @classmethod
    def get_loader(cls):

        env = settings.ENV

        if cls._loader is not None:
            return cls._loader

        if env == 'development':
            models_dir = os.getenv("MODELS_DIR")
            cls._loader = LocalModelLoader(models_dir)
        else:
            raise NotImplementedError(f"Loader for {env} not implemented")

        return cls._loader
