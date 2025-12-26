"""
Simple VLM wrapper for Parallax inspired by Apple's MLXVLM architecture.

This module provides a clean interface for Vision Language Models using mlx-vlm,
designed to integrate with Parallax's existing model architecture while maintaining
simplicity and avoiding distributed complexity for vision processing.
"""

from typing import Dict, List, Optional, Union

try:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    MLX_VLM_AVAILABLE = True
except ImportError:
    MLX_VLM_AVAILABLE = False

from parallax_utils.logging_config import get_logger

logger = get_logger("parallax.models.vlm")


class VLMModelError(Exception):
    """Exception raised for VLM model related errors."""

    pass


class ParallaxVLM:
    """
    Vision Language Model wrapper for Parallax.

    Provides a simplified interface for VLM inference using mlx-vlm,
    designed to work alongside Parallax's existing LLM infrastructure.
    """

    def __init__(self, model_path: str):
        """
        Initialize VLM with model path.

        Args:
            model_path: Path or Hugging Face model ID for the VLM model
        """
        if not MLX_VLM_AVAILABLE:
            raise VLMModelError(
                "mlx-vlm is not installed. Please install it with: pip install mlx-vlm"
            )

        self.model_path = model_path
        self.model = None
        self.processor = None
        self.config = None
        self._loaded = False

        logger.info(f"Initializing VLM wrapper for model: {model_path}")

    def load_model(self) -> None:
        """Load the VLM model and processor."""
        try:
            logger.info(f"Loading VLM model: {self.model_path}")
            self.model, self.processor = load(self.model_path)
            self.config = load_config(self.model_path)
            self._loaded = True
            logger.info("VLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            raise VLMModelError(f"Failed to load model {self.model_path}: {e}")

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded

    def generate_response(
        self,
        prompt: str,
        images: Optional[Union[str, List[str]]] = None,
        audio: Optional[Union[str, List[str]]] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
        verbose: bool = False,
    ) -> str:
        """
        Generate response from VLM.

        Args:
            prompt: Text prompt for the model
            images: Path(s) to image file(s) or URLs
            audio: Path(s) to audio file(s) (for supported models)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            verbose: Enable verbose output

        Returns:
            Generated text response
        """
        if not self._loaded:
            self.load_model()

        try:
            # Convert single items to lists
            if images is not None and isinstance(images, str):
                images = [images]
            if audio is not None and isinstance(audio, str):
                audio = [audio]

            # Apply chat template
            formatted_prompt = apply_chat_template(
                self.processor,
                self.config,
                prompt,
                num_images=len(images) if images else 0,
                num_audios=len(audio) if audio else 0,
            )

            output = generate(
                self.model,
                self.processor,
                formatted_prompt,
                images,
                audio=audio,
                max_tokens=max_tokens,
                verbose=False,
                temp=temperature,
            )

            return output

        except Exception as e:
            logger.error(f"VLM generation failed: {e}")
            raise VLMModelError(f"Generation failed: {e}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        images: Optional[Union[str, List[str]]] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """
        OpenAI-style chat completion interface.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            images: Path(s) to image file(s)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response text
        """
        # Extract the last user message as the prompt
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            raise VLMModelError("No user message found in chat completion request")

        prompt = user_messages[-1]["content"]

        return self.generate_response(
            prompt=prompt, images=images, max_tokens=max_tokens, temperature=temperature
        )

    @classmethod
    def get_architecture(cls) -> str:
        """Get the architecture name for the VLM wrapper."""
        return "VLMWrapper"

    @classmethod
    def get_popular_models(cls) -> List[str]:
        """Get list of popular VLM models that work well with this wrapper."""
        return [
            "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            "mlx-community/Qwen2.5-VL-32B-Instruct-8bit",
            "mlx-community/SmolVLM-Instruct-2.5B-4bit",
            "mlx-community/LLaVA-1.5-7B-4bit",
            "mlx-community/Idefics3-8B-4bit",
            "mlx-community/MiniCPM-V-2_6-4bit",
        ]


def create_vlm(model_path: str) -> ParallaxVLM:
    """
    Factory function to create a VLM instance.

    Args:
        model_path: Path or Hugging Face model ID

    Returns:
        ParallaxVLM instance
    """
    return ParallaxVLM(model_path)


EntryClass = ParallaxVLM
