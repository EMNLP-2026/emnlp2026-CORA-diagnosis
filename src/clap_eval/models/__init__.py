from .base import BaseClapModel
from .laion import LaionClapModel
from .mga import MGAClapModel
from .m2d import M2DClapModel
from .msclap import MSCLAPModel

def get_model(name: str, config: dict, device: str) -> BaseClapModel:
    if name.lower() == "laion":
        return LaionClapModel(name, config, device)
    elif name.lower() == "mga":
        return MGAClapModel(name, config, device)
    elif name.lower() == "m2d":
        return M2DClapModel(name, config, device)
    elif name.lower() == "msclap":
        return MSCLAPModel(name, config, device)
    else:
        raise ValueError(f"Unknown model name: {name}")