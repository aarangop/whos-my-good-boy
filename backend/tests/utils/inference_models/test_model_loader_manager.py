import os
import pytest
from unittest.mock import patch, MagicMock

from app.utils.inference_models.model_loader_manager import ModelLoaderManager


class TestModelLoaderManager:

    @patch("app.utils.inference_models.model_loader_manager.config")
    @patch('app.utils.inference_models.model_loader_manager.os.getenv')
    def test_get_loader_development_environment(self, mock_getenv, mock_config):
        """Test getting loader for local model"""
        # Arrange
        mock_config.MODEL_SOURCE = "local"
        mock_getenv.side_effect = lambda key, default=None: {
            "MODELS_DIR": "tests/test_data/models"
        }.get(key, default)

        with patch('app.utils.inference_models.model_loader_manager.LocalModelLoader') as mock_loader:
            # Create a mock instance
            mock_instance = MagicMock()
            mock_loader.return_value = mock_instance

            # Reset the singleton instance
            ModelLoaderManager._loader = None

            # Act
            loader = ModelLoaderManager.get_loader()

            # Assert
            mock_loader.assert_called_once_with("tests/test_data/models")
            assert loader == mock_instance

    @patch("app.utils.inference_models.model_loader_manager.S3ModelLoader")
    @patch("app.utils.inference_models.model_loader_manager.config")
    def test_get_loader_from_s3(self, mock_config, mock_s3_model_loader):
        """Test that getting an s3 model loader"""
        # Arrange
        mock_config.MODEL_SOURCE = "s3"
        mock_s3_model_loader.return_value = MagicMock()

        # Reset the singleton instance
        ModelLoaderManager._loader = None

        # Act & Assert
        ModelLoaderManager.get_loader()

    @patch("app.utils.inference_models.model_loader_manager.config")
    def test_get_loader_singleton_pattern(self, mock_config):
        """Test that get_loader returns the same instance when called multiple times"""
        # Arrange
        mock_config.MODEL_SOURCE = 'local'

        with patch('app.utils.inference_models.model_loader_manager.LocalModelLoader') as mock_loader:
            # Create a mock instance
            mock_instance = MagicMock()
            mock_loader.return_value = mock_instance

            # Reset the singleton instance
            ModelLoaderManager._loader = None

            # Act
            first_call = ModelLoaderManager.get_loader()
            second_call = ModelLoaderManager.get_loader()

            # Assert
            mock_loader.assert_called_once()  # Should only be instantiated once
            assert first_call == second_call

    @patch("app.core.config.os.getenv")
    def test_get_loader_default_environment(self, mock_getenv):
        """Test getting loader when ENVIRONMENT is not set"""
        # Arrange
        mock_getenv.side_effect = lambda key, default=None: {
            "MODELS_DIR": "tests/test_data/models"
        }.get(key, default)

        with patch('app.utils.inference_models.model_loader_manager.LocalModelLoader') as mock_loader:
            # Create a mock instance
            mock_instance = MagicMock()
            mock_loader.return_value = mock_instance

            # Reset the singleton instance
            ModelLoaderManager._loader = None

            # Act
            loader = ModelLoaderManager.get_loader()

            # Assert
            mock_loader.assert_called_once_with("tests/test_data/models")
            assert loader == mock_instance
