import os
import pytest
import tensorflow as tf
from unittest.mock import patch, MagicMock

from app.core import config
from app.utils.inference_models.model_loader import LocalModelLoader, S3ModelLoader
from app.utils.inference_models.model_loader_manager import ModelLoaderManager


class TestLocalModelLoader:

    def test_init_with_existing_dir(self, tmp_path):
        """Test initialization with an existing directory"""
        # Arrange
        models_dir = tmp_path

        # Act
        loader = LocalModelLoader(models_dir=str(models_dir))

        # Assert
        assert loader.models_dir == str(models_dir)
        assert loader.models == {}

    def test_init_with_non_existing_dir(self):
        """Test initialization with a non-existing directory raises error"""
        # Arrange
        non_existing_dir = "/path/does/not/exist"

        # Act & Assert
        with pytest.raises(ValueError, match=f"Models directory '{non_existing_dir}' does not exist"):
            LocalModelLoader(models_dir=non_existing_dir)

    def test_load_existing_model(self, tmp_path):
        """Test loading a model that exists on disk"""
        # Arrange
        models_dir = tmp_path
        model_filename = "test_model"
        model_path = os.path.join(models_dir, model_filename)

        loader = LocalModelLoader(models_dir=str(models_dir))

        # Create a mock file to simulate a model file
        with open(model_path, 'w') as f:
            f.write("dummy model content")

        # Act
        with patch('tensorflow.keras.models.load_model') as mock_load_model:
            mock_model = MagicMock(spec=tf.keras.models.Model)
            mock_load_model.return_value = mock_model

            # Mock os.path.join to monitor how it's called
            with patch('os.path.join', return_value=model_path) as mock_join:
                result = loader.load(model_filename)

                # Assert
                mock_join.assert_called_once_with(
                    str(models_dir), model_filename)
                mock_load_model.assert_called_once()
                assert result == mock_model
                assert model_filename in loader.models

    def test_load_non_existing_model(self, tmp_path):
        """Test loading a model that doesn't exist raises FileNotFoundError"""
        # Arrange
        models_dir = tmp_path
        model_filename = "non_existing_model"

        loader = LocalModelLoader(models_dir=str(models_dir))

        # Act & Assert
        with pytest.raises(FileNotFoundError, match=f"No model file '{os.path.join(str(models_dir), model_filename)}'"):
            loader.load(model_filename)

    def test_load_cached_model(self, tmp_path):
        """Test that a previously loaded model is returned from cache without reloading"""
        # Arrange
        models_dir = tmp_path
        model_filename = "test_model"
        model_path = os.path.join(models_dir, model_filename)

        # Create a mock file
        with open(model_path, 'w') as f:
            f.write("dummy model content")

        loader = LocalModelLoader(models_dir=str(models_dir))

        # Act
        with patch('tensorflow.keras.models.load_model') as mock_load_model:
            mock_model = MagicMock(spec=tf.keras.models.Model)
            mock_load_model.return_value = mock_model

            # First load
            first_result = loader.load(model_filename)

            # Second load - should use cache
            second_result = loader.load(model_filename)

            # Assert
            assert mock_load_model.call_count == 1  # Should only be called once
            assert first_result == second_result
            assert loader.models[model_filename] == mock_model

    def test_is_model_available(self, tmp_path):
        """Test is_model_available returns correct boolean based on model status"""
        # Arrange
        models_dir = tmp_path
        model_filename = "test_model"
        model_path = os.path.join(models_dir, model_filename)

        # Create a mock file
        with open(model_path, 'w') as f:
            f.write("dummy model content")

        loader = LocalModelLoader(models_dir=str(models_dir))

        # Act - Check before loading any models
        result_before = loader.is_model_available()

        # Load a model
        with patch('tensorflow.keras.models.load_model') as mock_load_model:
            mock_model = MagicMock(spec=tf.keras.models.Model)
            mock_load_model.return_value = mock_model
            loader.load(model_filename)

        # Act - Check after loading a model
        result_after = loader.is_model_available()

        # Assert
        assert result_before is False
        assert result_after is True

    def test_is_model_available_with_invalid_model(self, tmp_path):
        """Test is_model_available returns False if a model is not a valid TF model"""
        # Arrange
        models_dir = tmp_path
        model_filename = "test_model"
        model_path = os.path.join(models_dir, model_filename)

        # Create a mock file
        with open(model_path, 'w') as f:
            f.write("dummy model content")

        loader = LocalModelLoader(models_dir=str(models_dir))

        # Manually set an invalid model
        loader.models[model_filename] = "not a model"

        # Act
        result = loader.is_model_available()

        # Assert
        assert result is False


@pytest.mark.integration
class TestS3ModelLoader:

    def test_s3_model_loader_init(self):
        """Test S3ModelLoader initialization"""

        # Act
        loader = S3ModelLoader()

        # Assert
        assert loader._client is not None

    def test_s3_model_loader_loads_model_version(self):

        # Arrange
        loader = S3ModelLoader()

        # Act
        loader.load("cat-dog-other-classifier", version="v1")

        assert len(loader.models) > 0
        assert 'cat-dog-other-classifier' in loader.models
        assert isinstance(
            loader.models['cat-dog-other-classifier'], tf.keras.models.Model)

    def test_s3_model_loader_loads_model_latest(self):
        # Arrange
        loader = S3ModelLoader()

        # Act
        loader.load("cat-dog-other-classifier", version="latest")

        assert len(loader.models) > 0
        assert 'cat-dog-other-classifier' in loader.models
        assert isinstance(
            loader.models['cat-dog-other-classifier'], tf.keras.models.Model)

    def test_s3_model_loader_invalid_model(self):
        # Arrange
        loader = S3ModelLoader()

        # Act & Assert
        with pytest.raises(FileNotFoundError, match=f"Model 'non-existing-model' not found in S3"):
            loader.load("non-existing-model")

    def test_s3_model_loader_stores_model_in_local_cache(self, tmp_path):
        # Arrange
        loader = S3ModelLoader()
        model_filename = "cat-dog-other-classifier"
        version = "latest"
        local_cache_dir = tmp_path / "s3_model_cache"
        local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Act
        loader.load(model_filename, version=version,
                    local_cache_dir=str(local_cache_dir))

        # Assert
        assert len(loader.models) > 0
        assert model_filename in loader.models
        assert isinstance(loader.models[model_filename], tf.keras.models.Model)
        # Check if the model is stored in the local cache
        cached_model_path = os.path.join(
            str(local_cache_dir), f"{model_filename}", "v1", "model.h5")
        assert os.path.exists(cached_model_path)
