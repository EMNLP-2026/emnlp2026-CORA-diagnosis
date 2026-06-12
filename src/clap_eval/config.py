import yaml
from pathlib import Path

class Config:
    def __init__(self, config_dict: dict):
        self._config = config_dict

    @classmethod
    def load(cls, config_path: str | Path):
        with open(config_path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    @property
    def dataset(self) -> dict:
        return self._config.get("dataset", {})

    @property
    def models(self) -> list:
        return self._config.get("models", [])

    @property
    def execution(self) -> dict:
        return self._config.get("execution", {})

    @property
    def evaluation(self) -> dict:
        return self._config.get("evaluation", {})

    @property
    def analysis(self) -> dict:
        return self._config.get("analysis", {})