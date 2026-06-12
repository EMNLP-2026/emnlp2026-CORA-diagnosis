import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
import librosa
from .base import BaseClapModel

class LaionClapModel(BaseClapModel):
    def _load_model(self):
        source = self.model_path or self.hf_model_id
        if not source:
            raise ValueError("LAION config must set model_path to a local Transformers model directory or cache id.")
        print(f"Loading '{self.name}' from local Transformers source: {source} on {self.device}")
        self.model = ClapModel.from_pretrained(source, local_files_only=self.local_files_only).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(source, local_files_only=self.local_files_only)
        self.resampler_cache = {}
        
    @torch.no_grad()
    def get_audio_embedding(self, audio_data: list[np.ndarray], sr: int) -> np.ndarray:
        processed_audios = []
        for arr in audio_data:
            if len(arr.shape) > 1 and arr.shape[0] > 1:
                arr = arr.mean(axis=0)
            processed_audios.append(arr)

        # Standard LAION CLAP sample rate is mostly 48kHz
        target_sr = self.processor.feature_extractor.sampling_rate
        
        if sr != target_sr:
            import torchaudio.transforms as T
            if sr not in self.resampler_cache:
                self.resampler_cache[sr] = T.Resample(orig_freq=sr, new_freq=target_sr).to(self.device)
                
            max_len = max(len(a) for a in processed_audios) if processed_audios else 0
            padded = np.zeros((len(processed_audios), max_len), dtype=np.float32)
            for i, a in enumerate(processed_audios):
                padded[i, :len(a)] = a
                
            audio_tensor = torch.from_numpy(padded).float().to(self.device)
            audio_tensor = self.resampler_cache[sr](audio_tensor)
            audio_batch = list(audio_tensor.cpu().numpy())
            inputs = self.processor(audio=audio_batch, sampling_rate=target_sr, return_tensors="pt")
        else:
            inputs = self.processor(audio=processed_audios, sampling_rate=target_sr, return_tensors="pt")
        
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, 'to')}
        
        outputs = self.model.get_audio_features(**inputs)
        # HF ClapModel get_audio_features sometimes returns BaseModelOutputWithPooling 
        # where the projected feature is in pooler_output
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, "audio_features"):
            embeddings = outputs.audio_features
        elif isinstance(outputs, tuple):
            embeddings = outputs[0]
        else:
            embeddings = outputs
            
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, 'to')}
        
        outputs = self.model.get_text_features(**inputs)
        # Similar to audio features, might return BaseModelOutputWithPooling
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, "text_features"):
            embeddings = outputs.text_features
        elif hasattr(outputs, "text_embeds"):
            embeddings = outputs.text_embeds
        elif isinstance(outputs, tuple):
            embeddings = outputs[0]
        else:
            embeddings = outputs
            
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()