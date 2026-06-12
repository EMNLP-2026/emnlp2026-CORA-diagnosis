import abc
import numpy as np

class BaseClapModel(abc.ABC):
    """
    Abstract base class for CLAP wrappers (LAION, MGA, M2D).
    All models should implement this interface to be integrated into the test pipeline.
    """
    def __init__(self, name: str, config: dict, device: str):
        self.name = name
        self.config = config
        self.device = device

        self.hf_model_id = config.get("hf_model_id")
        self.model_path = config.get("model_path") or config.get("local_model_path")
        self.local_files_only = bool(config.get("local_files_only", True))
        self.repo_path = config.get("repo_path")
        self.checkpoint_path = config.get("checkpoint_path")

        self._load_model()

    @abc.abstractmethod
    def _load_model(self):
        """Load model weights and configs from explicit local paths."""
        pass

    @abc.abstractmethod
    def get_audio_embedding(self, audio_data: list[np.ndarray], sr: int) -> np.ndarray:
        """
        Takes a list of raw audio signals (numpy arrays) and sample rate, and outputs the audio embeddings.
        Must handle its own required sequence length, resampling, or padding across the batch.
        """
        pass

    @abc.abstractmethod
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        """
        Takes a list of text string inputs and outputs the text embeddings.
        """
        pass