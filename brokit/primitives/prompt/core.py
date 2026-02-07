from typing import Dict
from brokit.primitives.prompt.types import FieldInfo

class PromptMeta(type):
    # Add type hints for metaclass attributes
    _input_fields: Dict[str, FieldInfo]
    _output_fields: Dict[str, FieldInfo]
    _instructions: str    
    def __new__(cls, name, bases, namespace):
        # Check if this is from from_dict (already processed)
        if '_input_fields' in namespace and '_output_fields' in namespace and '_instructions' in namespace:
            # Already processed, just create the class
            return super().__new__(cls, name, bases, namespace)        
        input_fields = {}
        output_fields = {}
        # Get type annotations
        annotations = namespace.get('__annotations__', {})
        for field_name, field_value in list(namespace.items()):
            if isinstance(field_value, FieldInfo):
                field_value.name = field_name
                
                # Extract type from annotation
                if field_name in annotations:
                    field_value.type = annotations[field_name]
                
                if field_value.is_input:
                    input_fields[field_name] = field_value
                else:
                    output_fields[field_name] = field_value
                
                del namespace[field_name]
        
        namespace['_input_fields'] = input_fields
        namespace['_output_fields'] = output_fields
        namespace['_instructions'] = (namespace.get('__doc__', '') or '').strip()
        return super().__new__(cls, name, bases, namespace)
    
    @property
    def input_fields(cls):
        return cls._input_fields
    
    @property
    def output_fields(cls):
        return cls._output_fields
    
    @property
    def instructions(cls):
        return cls._instructions

class Prompt(metaclass=PromptMeta):
    
    """Base class for defining prompts with input and output fields."""
    # Type hints for class attributes set by metaclass
    _input_fields: Dict[str, FieldInfo]
    _output_fields: Dict[str, FieldInfo]
    _instructions: str     

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            if name in self._input_fields or name in self._output_fields:
                setattr(self, name, value)
            else:
                raise ValueError(f"Unknown field: {name}")
        
        for name in self._input_fields:
            if name not in kwargs:
                raise ValueError(f"Missing required input: {name}")
    
    def __getattr__(self, name):
        if name in self._output_fields:
            return "Intentionally left blank."
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    
    @property
    def inputs(self):
        """Get all input values (X)."""
        return {name: getattr(self, name) for name in self._input_fields}
    
    @property
    def outputs(self):
        """Get all output values (y)."""
        return {name: getattr(self, name) for name in self._output_fields 
                if hasattr(self, name)}
    
    def is_complete(self):
        """Check if all outputs are provided."""
        return len(self.outputs) == len(self._output_fields)    