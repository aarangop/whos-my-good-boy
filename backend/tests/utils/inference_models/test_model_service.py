import os
import pytest
from unittest.mock import patch, MagicMock

from app.utils.inference_models.model_loader_manager import ModelLoaderManager


class TestModelLoaderManager:

    @patch("os.getenv")
    def test_get_loader_development_environment(self, mock_getenv):
        """Test getting loader in development environment"""
        # Arrange
        mock_getenv.side_effect = lambda key, default=None: {
            "ENVIRONMENT": "development",
            "MODELS_DIR": "tests/test_data/models"
        }.get(key, default)

        with patch('app.utils.inference_models.model_loader.LocalModelLoader') as mock_loader:
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

    @patch("os.getenv")
    def test_get_loader_non_development_environment(self, mock_getenv):
        """Test that getting loader in non-development environment raises exception"""
        # Arrange
        mock_getenv.side_effect = lambda key, default=None: {
            "ENVIRONMENT": "production"
        }.get(key, default)

        # Reset the singleton instance
        ModelLoaderManager._loader = None

        # Act & Assert
        with pytest.raises(NotImplementedError, match="Loader for production not implemented"):
            ModelLoaderManager.get_loader()

    @patch("os.getenv")
    def test_get_loader_singleton_pattern(self, mock_getenv):
        """Test that get_loader returns the same instance when called multiple times"""
        # Arrange
        mock_getenv.side_effect = lambda key, default=None: {
            "ENVIRONMENT": "development",
            "MODELS_DIR": "tests/test_data/models"
        }.get(key, default)

        with patch('app.utils.inference_models.model_loader.LocalModelLoader') as mock_loader:
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

    @patch("os.getenv")
    def test_get_loader_default_environment(self, mock_getenv):
        """Test getting loader when ENVIRONMENT is not set"""
        # Arrange
        mock_getenv.side_effect = lambda key, default=None: {
            "MODELS_DIR": "tests/test_data/models"
        }.get(key, default)

        with patch('app.utils.inference_models.model_loader.LocalModelLoader') as mock_loader:
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
