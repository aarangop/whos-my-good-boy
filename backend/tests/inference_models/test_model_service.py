import os
import pytest
from unittest.mock import patch, MagicMock

from app.inference_models.model_service import ModelLoaderManager
from app.inference_models.model_loader import LocalModelLoader


class TestModelLoaderManager:

    def test_get_loader_development_environment(self):
        """Test getting loader in development environment"""
        # Arrange
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "MODELS_DIR": "/test/models"}):
            with patch('app.inference_models.model_service.LocalModelLoader') as mock_loader:
                # Create a mock instance
                mock_instance = MagicMock()
                mock_loader.return_value = mock_instance

                # Reset the singleton instance
                ModelLoaderManager._loader = None

                # Act
                loader = ModelLoaderManager.get_loader()

                # Assert
                mock_loader.assert_called_once_with("/test/models")
                assert loader == mock_instance

    def test_get_loader_non_development_environment(self):
        """Test that getting loader in non-development environment raises exception"""
        # Arrange
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            # Reset the singleton instance
            ModelLoaderManager._loader = None

            # Act & Assert
            with pytest.raises(NotImplementedError, match="Loader for production not implemented"):
                ModelLoaderManager.get_loader()

    def test_get_loader_singleton_pattern(self):
        """Test that get_loader returns the same instance when called multiple times"""
        # Arrange
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "MODELS_DIR": "/test/models"}):
            with patch('app.inference_models.model_service.LocalModelLoader') as mock_loader:
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

    def test_get_loader_default_environment(self):
        """Test getting loader when ENVIRONMENT is not set"""
        # Arrange
        # Ensure ENVIRONMENT is not set, but set MODELS_DIR
        with patch.dict(os.environ, {"MODELS_DIR": "/test/models"}, clear=True):
            with patch('app.inference_models.model_service.LocalModelLoader') as mock_loader:
                # Create a mock instance
                mock_instance = MagicMock()
                mock_loader.return_value = mock_instance

                # Reset the singleton instance
                ModelLoaderManager._loader = None

                # Act
                loader = ModelLoaderManager.get_loader()

                # Assert
                mock_loader.assert_called_once_with("/test/models")
                assert loader == mock_instance
