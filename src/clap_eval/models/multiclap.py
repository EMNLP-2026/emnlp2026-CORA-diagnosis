import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import Optional

class MultiCLAP:
    def __init__(self, models_dict: dict, device='cuda'):
        self.models = models_dict
        # [수정됨] 고정된 이름이 아니라, 현재 로드된(켜진) 모델들만 자동으로 리스트업합니다!
        self.model_names = list(models_dict.keys())
        self.device = device
        
        print(f"MultiCLAP initialized with {len(self.model_names)} models: {self.model_names}")

    def _get_all_similarities(self, audio_data: list[np.ndarray], sr: int, texts: list[str]) -> dict:
        sim_matrices = {}
        for name in self.model_names:
            model = self.models[name]
            
            audio_emb = model.get_audio_embedding(audio_data, sr)
            text_emb = model.get_text_embedding(texts)
            
            audio_tensor = torch.from_numpy(audio_emb).to(self.device)
            text_tensor = torch.from_numpy(text_emb).to(self.device)
            
            audio_tensor = F.normalize(audio_tensor, p=2, dim=-1)
            text_tensor = F.normalize(text_tensor, p=2, dim=-1)
            
            sim_matrix = audio_tensor @ text_tensor.T
            sim_matrices[name] = sim_matrix
            
        return sim_matrices

    def predict_zero_shot(
        self,
        audio_data: list[np.ndarray],
        sr: int,
        text_prompts: list[str],
        class_names: Optional[list[str]] = None,
    ) -> list[str]:
        if class_names is None:
            class_names = text_prompts
        if len(text_prompts) != len(class_names):
            raise ValueError(
                f"text_prompts and class_names must have the same length: "
                f"{len(text_prompts)} != {len(class_names)}"
            )

        sim_matrices = self._get_all_similarities(audio_data, sr, text_prompts)
        batch_size = len(audio_data)
        final_predictions = []

        for i in range(batch_size):
            votes = {}
            for name in self.model_names:
                pred_idx = torch.argmax(sim_matrices[name][i]).item()
                votes[name] = pred_idx

            vote_list = list(votes.values())
            counts = Counter(vote_list)
            max_votes = max(counts.values())
            candidates = [idx for idx, count in counts.items() if count == max_votes]

            # [수정됨] 동점 처리 안전장치 (m2d가 꺼져있을 경우 대비)
            if len(candidates) > 1:
                if "m2d" in votes and votes["m2d"] in candidates:
                    final_pred_idx = votes["m2d"]
                else:
                    final_pred_idx = candidates[0]
            else:
                final_pred_idx = candidates[0]

            final_predictions.append(class_names[final_pred_idx])

        return final_predictions

    def get_ensemble_similarity_matrix(self, audio_data: list[np.ndarray], sr: int, texts: list[str]) -> np.ndarray:
        sim_matrices = self._get_all_similarities(audio_data, sr, texts)
        
        normalized_matrices = []
        for name, matrix in sim_matrices.items():
            min_val = matrix.min()
            max_val = matrix.max()
            # 분모가 0이 되는 것을 방지하기 위한 clamp
            norm_matrix = (matrix - min_val) / torch.clamp((max_val - min_val), min=1e-8)
            normalized_matrices.append(norm_matrix)
            
        stacked = torch.stack(normalized_matrices)
        ensemble_matrix = torch.mean(stacked, dim=0)
        
        return ensemble_matrix.cpu().numpy()
