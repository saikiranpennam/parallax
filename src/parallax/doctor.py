import os
import re

import torch

from parallax_utils.logging_config import get_logger

logger = get_logger("parallax.doctor")


def check_backend():
    """
    Check GPU availability for CUDA (NVIDIA) and Metal (Apple Silicon)
    """
    try:
        cuda_available = torch.cuda.is_available()

        # Check for CUDA
        if cuda_available:
            logger.info("CUDA Available")
            logger.info(f" CUDA Version: {torch.version.cuda}")
            logger.info(f" Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f" GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(
                    f" Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
                )
            logger.info("CUDA Available")

        # Check for Metal (MPS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Metal Performance Shaders (MPS) Available")
            logger.info("MPS backend available")

        else:
            # No GPU found
            logger.info("No GPU acceleration available")
            logger.info("No CUDA or Metal devices detected")
            logger.info("No GPU available")

    except ImportError:
        logger.warning("Pytorch not installed")
    except Exception as e:
        logger.exception(f"Error checking GPU: {str(e)}")


def check_dependencies():
    """Check required package dependencies from pyproject.toml or requirements.txt"""
    dependencies = []
    dep_source = None

    # Try to read from pyproject.toml first
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "r") as f:
            content = f.read()
            dependencies = re.findall(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
            if dependencies:
                dep_source = "pyproject.toml"

    # If pyproject.toml is not found, try requirements.txt
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            content = f.read()
            dependencies = content.splitlines()
            if dependencies:
                dep_source = "requirements.txt"
    else:
        logger.warning("No dependency file found")

    if dependencies:
        logger.info(f"Dependencies found in {dep_source}")
        for dep in dependencies:
            logger.info(f"Dependency: {dep}")
            # print(f"Dependency: {dep}")
    else:
        logger.warning("No dependencies found")


if __name__ == "__main__":
    try:
        check_backend()
        # check_dependencies()
    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(e)
