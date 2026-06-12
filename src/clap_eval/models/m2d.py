import torch
import numpy as np
import sys
from pathlib import Path
from .base import BaseClapModel

class M2DClapModel(BaseClapModel):
    def _load_model(self):
        print(f"Loading '{self.name}' from local path: {self.checkpoint_path} on {self.device}")
        
        # Add the external repository to Python's sys.path dynamically
        if self.repo_path:
            # We need to add the examples directory because portable_m2d.py is there
            examples_dir = str((Path(self.repo_path) / "examples").resolve())
            if examples_dir not in sys.path:
                sys.path.insert(0, examples_dir)
                print(f"Added {examples_dir} to sys.path")
            
        try:
            # PortableM2D parses directory name (e.g. m2d_clap_vit_base-80x1001p16x16...)
            # If our checkpoint is just in "checkpoints", we must symlink it to a dummy well-formed directory.
            ckpt_path_obj = Path(self.checkpoint_path)
            if "-" not in ckpt_path_obj.parent.name:
                simulated_dir = ckpt_path_obj.parent / "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
                simulated_dir.mkdir(parents=True, exist_ok=True)
                simulated_path = simulated_dir / ckpt_path_obj.name
                if not simulated_path.exists():
                    import shutil
                    try:
                        simulated_path.symlink_to(ckpt_path_obj.resolve())
                    except OSError:
                        shutil.copy(ckpt_path_obj, simulated_path)
                weight_file = str(simulated_path)
                print(f"Forwarded M2D checkpoint to {weight_file} for proper parsing")
            else:
                weight_file = self.checkpoint_path

            from portable_m2d import PortableM2D
            # Use flat_features=True for CLAP model 
            self.model = PortableM2D(weight_file=weight_file, flat_features=True)
            
            # Entire model (including to_spec, backbone, projections) must go to device
            self.model.to(self.device)
            self.model.eval()
            
        except ImportError as e:
            print(f"Failed to import M2D model code: {e}")
            raise e
        except Exception as e:
            print(f"Failed to load M2D model: {e}")
            raise e

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: list[np.ndarray], sr: int) -> np.ndarray:
        target_sr = 16000
        target_length = 16000 * 10
        
        processed_audios = []
        for arr in audio_data:
            if len(arr.shape) > 1 and arr.shape[0] > 1:
                arr = arr.mean(axis=0)
            processed_audios.append(arr)

        max_len = max(len(a) for a in processed_audios) if processed_audios else 0
        padded = np.zeros((len(processed_audios), max_len), dtype=np.float32)
        for i, a in enumerate(processed_audios):
            padded[i, :len(a)] = a
            
        audio_tensor = torch.from_numpy(padded).float()

        if sr != target_sr:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            audio_tensor = resampler(audio_tensor)
            
        # Ensure it's 10-seconds (160000 samples) as M2D CLAP usually expects exactly 10s audio
        if audio_tensor.shape[-1] > target_length:
            audio_tensor = audio_tensor[:, :target_length]
        elif audio_tensor.shape[-1] < target_length:
            import torch.nn.functional as F
            audio_tensor = F.pad(audio_tensor, (0, target_length - audio_tensor.shape[-1]))
            
        audio_tensor = audio_tensor.to(self.device, non_blocking=True)
            
        embs = self.model.encode_clap_audio(audio_tensor)
        return embs.cpu().numpy()

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(texts), 768))
        embs = self.model.encode_clap_text(texts)
        if isinstance(embs, torch.Tensor):
            embs = embs.to(self.device)
        return embs.cpu().numpy()