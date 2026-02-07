from typing import Any

class Prediction:
    def __init__(self, **kwargs: Any) -> None:
        self._data = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self) -> str:
        items = ",\n    ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"Prediction(\n    {items}\n)"
    
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def to_dict(self) -> dict:
        return self._data        
