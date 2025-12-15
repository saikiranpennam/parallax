"""Test suite for parallax doctor"""

from unittest.mock import Mock, patch

import pytest

from parallax.doctor import check_backend


class TestCheckBackend:
    """Test suite for check_backend function"""

    @patch("parallax.doctor.torch.cuda.is_available")
    @patch("parallax.doctor.torch.cuda.device_count")
    @patch("parallax.doctor.torch.cuda.get_device_name")
    @patch("parallax.doctor.torch.cuda.get_device_properties")
    def test_cuda_available_single_gpu(
        self, mock_props, mock_name, mock_count, mock_available, logger
    ):
        """Test when CUDA is available with a single GPU"""
        # Setup mocks
        mock_available.return_value = True
        mock_count.return_value = 1
        mock_name.return_value = "NVIDIA GeForce RTX 3090"

        mock_device_props = Mock()
        mock_device_props.total_memory = 24 * 1e9  # 24 GB
        mock_props.return_value = mock_device_props

        with patch("torch.version.cuda", "11.8"):
            check_backend()

        # Verify logging calls
        assert logger.info.call_count >= 5
        logger.info.assert_any_call("CUDA Available")
        logger.info.assert_any_call("CUDA Version: 11.8")
        logger.info.assert_any_call("Device Count: 1")
        logger.info.assert_any_call("GPU 0: NVIDIA GeForce RTX 3090")

    @patch("parallax.doctor.torch.cuda.is_available")
    @patch("parallax.doctor.torch.cuda.device_count")
    @patch("parallax.doctor.torch.cuda.get_device_name")
    @patch("parallax.doctor.torch.cuda.get_device_properties")
    def test_cuda_available_multiple_gpus(
        self, mock_props, mock_name, mock_count, mock_available, logger
    ):
        """Test when CUDA is available with multiple GPUs"""
        mock_available.return_value = True
        mock_count.return_value = 2
        mock_name.side_effect = ["NVIDIA A100", "NVIDIA A100"]

        mock_device_props = Mock()
        mock_device_props.total_memory = 40 * 1e9  # 40 GB
        mock_props.return_value = mock_device_props

        with patch("torch.version.cuda", "12.0"):
            check_backend()

        # Verify both GPUs are logged
        logger.info.assert_any_call("GPU 0: NVIDIA A100")
        logger.info.assert_any_call("GPU 1: NVIDIA A100")
        assert mock_name.call_count == 2
        assert mock_props.call_count == 2

    @patch("parallax.doctor.torch.cuda.is_available")
    @patch("parallax.doctor.torch.backends.mps.is_available")
    def test_mps_available(self, mock_mps_available, mock_cuda_available, logger):
        """Test when Metal Performance Shaders (MPS) is available"""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True

        with patch("torch.backends", create=True) as mock_backends:
            mock_backends.mps.is_available.return_value = True
            check_backend()

        logger.info.assert_any_call("Metal Performance Shaders (MPS) Available")
        logger.info.assert_any_call("MPS backend available")

    @patch("parallax.doctor.torch.cuda.is_available")
    def test_no_gpu_available(self, mock_available, logger):
        """Test when no GPU acceleration is available"""
        mock_available.return_value = False

        # Mock torch.backends without mps attribute
        with patch("torch.backends", spec=[]):
            check_backend()

        logger.info.assert_any_call("No GPU acceleration available")
        logger.info.assert_any_call("No CUDA or Metal devices detected")
        logger.info.assert_any_call("No GPU available")

    @patch("parallax.doctor.torch.cuda.is_available")
    def test_mps_not_available(self, mock_available, logger):
        """Test when MPS backend exists but is not available"""
        mock_available.return_value = False

        with patch("torch.backends", create=True) as mock_backends:
            mock_backends.mps.is_available.return_value = False
            check_backend()

        logger.info.assert_any_call("No GPU acceleration available")

    @patch("parallax.doctor.torch.cuda.is_available")
    def test_exception_during_check(self, mock_available, logger):
        """Test exception handling during GPU check"""
        mock_available.side_effect = RuntimeError("CUDA error")

        check_backend()

        logger.exception.assert_called_once()
        assert "Error checking GPU:" in logger.exception.call_args[0][0]

    @patch("parallax.doctor.torch.cuda.is_available")
    @patch("parallax.doctor.torch.cuda.device_count")
    @patch("parallax.doctor.torch.cuda.get_device_name")
    def test_cuda_with_zero_devices(self, mock_name, mock_count, mock_available, logger):
        """Test when CUDA is available but no devices are detected"""
        mock_available.return_value = True
        mock_count.return_value = 0

        with patch("torch.version.cuda", "11.8"):
            check_backend()

        logger.info.assert_any_call("Device Count: 0")
        # Should not call get_device_name since no devices
        mock_name.assert_not_called()


@pytest.fixture
def logger():
    """Fixture to provide a mocked logger"""
    with patch("parallax.doctor.logger") as mock_logger:
        yield mock_logger
