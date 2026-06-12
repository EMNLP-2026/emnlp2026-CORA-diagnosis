import torch
import numpy as np
import sys
import yaml
from pathlib import Path
from .base import BaseClapModel


def _install_sed_scores_eval_compat() -> None:
    try:
        import sed_scores_eval.utils.scores  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        from sed_scores_eval.base_modules.scores import create_score_dataframe
        import sed_scores_eval.utils as utils_pkg
    except ModuleNotFoundError:
        return

    import types

    scores_module = types.ModuleType("sed_scores_eval.utils.scores")
    scores_module.create_score_dataframe = create_score_dataframe
    sys.modules["sed_scores_eval.utils.scores"] = scores_module
    setattr(utils_pkg, "scores", scores_module)


def _install_sparsemax_device_compat() -> None:
    try:
        from models.ase_model import Sparsemax
    except ModuleNotFoundError:
        return

    original_forward = Sparsemax.forward
    if getattr(original_forward, "_cora_device_compat", False):
        return

    def forward(self, input, device=None):
        if device is None or device == "cuda":
            device = input.device
        return original_forward(self, input, device=device)

    forward._cora_device_compat = True
    Sparsemax.forward = forward


class MGAClapModel(BaseClapModel):
    def _load_model(self):
        print(f"Loading '{self.name}' from local path: {self.checkpoint_path} on {self.device}")
        
        # Add the external repository to Python's sys.path dynamically
        repo_abs_path = None
        if self.repo_path:
            repo_abs_path = str(Path(self.repo_path).resolve())
            if repo_abs_path not in sys.path:
                sys.path.insert(0, repo_abs_path)
                print(f"Added {repo_abs_path} to sys.path")
        
        try:
            _install_sed_scores_eval_compat()
            from models.ase_model import ASE
            _install_sparsemax_device_compat()
            import torchaudio.transforms as T
            from ruamel.yaml import YAML

            # Load configuration file required by MGA-CLAP
            config_path = Path(repo_abs_path) / "settings" / "inference_example.yaml"
            with open(config_path, "r") as f:
                yaml = YAML(typ='safe', pure=True)
                config = yaml.load(f)

            config["device"] = self.device
            self.model = ASE(config)
            self.model.to(self.device)

            # Load weights
            cp = torch.load(self.checkpoint_path, map_location=self.device,weights_only=False)
            # Some checkpoints dict load `model` key
            state_dict = cp['model'] if 'model' in cp else cp
            # strict=False 로 변경하여 사소한 키 불일치를 무시합니다.
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print(f"Model weights loaded from {self.checkpoint_path}")
            
            self.target_sr = config.get("audio_args", {}).get("sr", 32000)
            self.resampler_cache = {}

        except Exception as e:
            print(f"Failed to import/load MGA model code: {e}")
            raise e

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: list[np.ndarray], sr: int) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(audio_data), 512))

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

        if sr != self.target_sr:
            if sr not in self.resampler_cache:
                import torchaudio.transforms as T
                self.resampler_cache[sr] = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            audio_tensor = self.resampler_cache[sr](audio_tensor)
            
        max_length = 10 * self.target_sr
        if audio_tensor.shape[-1] > max_length:
            audio_tensor = audio_tensor[:, :max_length]
        
        audio_tensor = audio_tensor.to(self.device, non_blocking=True)

        _, frame_embeds = self.model.encode_audio(audio_tensor)
        audio_embeds = self.model.msc(frame_embeds, self.model.codebook)
        
        return audio_embeds.cpu().numpy()

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(texts), 512))
        
        _, word_embeds, attn_mask = self.model.encode_text(texts)
        text_embeds = self.model.msc(word_embeds, self.model.codebook, attn_mask)
        
        return text_embeds.cpu().numpy()