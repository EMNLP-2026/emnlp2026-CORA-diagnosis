import torch
import numpy as np
import torchaudio.transforms as T
from msclap import CLAP
from .base import BaseClapModel

class MSCLAPModel(BaseClapModel):
    def _load_model(self):
        version = self.config.get("version", "2023")
        print(f"Loading '{self.name}' (MSCLAP version {version}) on {self.device}")
        
        if not self.checkpoint_path:
            raise ValueError("MS-CLAP config must set checkpoint_path. Download the checkpoint manually and pass its local path.")
        ckpt = self.checkpoint_path
        
        # Accelerate returns a torch.device object, so we must check if "cuda" is in its string representation
        is_cuda = "cuda" in str(self.device)
        self.model = CLAP(ckpt, version=version, use_cuda=is_cuda)
        self.model.clap.eval()
        self.model.clap = self.model.clap.to(self.device)
        
        self.target_sr = self.model.args.sampling_rate
        self.duration = self.model.args.duration
        self.resampler_cache = {}

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: list[np.ndarray], sr: int) -> np.ndarray:
        processed_audios = []
        for arr in audio_data:
            if len(arr.shape) > 1 and arr.shape[0] > 1:
                arr = arr.mean(axis=0)
            processed_audios.append(arr)

        target_length = int(self.target_sr * self.duration)
        max_len = max(len(a) for a in processed_audios) if processed_audios else 0
        padded = np.zeros((len(processed_audios), max_len), dtype=np.float32)
        for i, a in enumerate(processed_audios):
            padded[i, :len(a)] = a

        audio_tensor = torch.from_numpy(padded).float()

        if sr != self.target_sr:
            if sr not in self.resampler_cache:
                self.resampler_cache[sr] = T.Resample(orig_freq=sr, new_freq=self.target_sr).to(self.device)
            audio_tensor = audio_tensor.to(self.device)
            audio_tensor = self.resampler_cache[sr](audio_tensor)
        else:
            audio_tensor = audio_tensor.to(self.device, non_blocking=True)

        if audio_tensor.shape[-1] > target_length:
            audio_tensor = audio_tensor[:, :target_length]
        elif audio_tensor.shape[-1] < target_length:
            import torch.nn.functional as F
            audio_tensor = F.pad(audio_tensor, (0, target_length - audio_tensor.shape[-1]))

        embeddings = self.model.clap.audio_encoder(audio_tensor)[0]
        
        if hasattr(torch.nn.functional, "normalize"):
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        preprocessed_text = self.model.preprocess_text(texts)
        if "cuda" in str(self.device):
            if hasattr(preprocessed_text, 'to'):
                preprocessed_text = preprocessed_text.to(self.device)
            elif isinstance(preprocessed_text, dict):
                # Using comprehension to build a new dict handles BatchEncoding wrapper cases
                preprocessed_text = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in preprocessed_text.items()}
            
        embeddings = self.model.clap.caption_encoder(preprocessed_text)
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        return embeddings