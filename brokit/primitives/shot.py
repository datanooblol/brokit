from brokit.primitives.prompt.core import Prompt
from typing import Optional, Type

class Shot:
    def __init__(self, prompt_class:Optional[Type[Prompt]]=None, **kwargs):
        """
        Create a shot with automatic input/output separation.
        
        Args:
            prompt_class: Optional Prompt class to validate fields
            **kwargs: Field values
        """
        self._prompt_class = prompt_class
        self._data = kwargs
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def inputs(self):
        """Get input fields based on prompt class."""
        if self._prompt_class:
            return {k: v for k, v in self._data.items() 
                    if k in self._prompt_class.input_fields}
        return {}
    
    @property
    def outputs(self):
        """Get output fields with defaults for missing ones."""
        if self._prompt_class:
            result = {}
            for field_name in self._prompt_class.output_fields.keys():
                if field_name in self._data:
                    result[field_name] = self._data[field_name]
                else:
                    result[field_name] = "Intentionally left blank."
            return result
        return {}
    
    def __getattr__(self, name):
        # Return default for missing output fields
        if self._prompt_class and name in self._prompt_class.output_fields:
            return "Intentionally left blank."
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    
    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"Shot({items})"
