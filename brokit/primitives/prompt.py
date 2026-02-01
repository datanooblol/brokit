"""
Prompt is where we keep the prompt contract for the program

What should Prompt do:
- construct new class from MetaClass concept
- in new class, have input_fields, output_fields, instructions as properties
- the class itself should be composable with/without output_fields
- Prompt is simply the way to construct the prompt which will be formatted later with Predictor
- Since prompt here has both input/output, I think we can make used of it as example or demo?
"""

from dataclasses import dataclass
from typing import Type, Any

@dataclass
class FieldInfo:
    name: str
    description: str
    type: Type
    is_input: bool
    
    def to_dict(self):
        """Serialize to JSON-compatible dict."""
        return {
            'name': self.name,
            'description': self.description,
            'type': self.type.__name__,  # 'str', 'int', etc.
            'is_input': self.is_input
        }
    
    @staticmethod
    def from_dict(data):
        """Deserialize from dict."""
        # Map type name back to actual type
        type_map = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
        }
        return FieldInfo(
            name=data['name'],
            description=data['description'],
            type=type_map.get(data['type'], str),
            is_input=data['is_input']
        )

def InputField(description:str = "") -> Any:
    return FieldInfo(name="", description=description, type=str, is_input=True)

def OutputField(description:str = "") -> Any:
    return FieldInfo(name="", description=description, type=str, is_input=False)

class PromptMeta(type):
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

    @classmethod
    def to_dict(cls):
        """Export to JSON-serializable dict."""
        return {
            'name': cls.__name__,
            'input_fields': {k: v.to_dict() for k, v in cls._input_fields.items()},
            'output_fields': {k: v.to_dict() for k, v in cls._output_fields.items()},
            'instructions': cls._instructions,
        }

    @classmethod
    def from_dict(cls, data):
        """Recreate from dict."""
        namespace = {
            '_input_fields': {k: FieldInfo.from_dict(v) for k, v in data['input_fields'].items()},
            '_output_fields': {k: FieldInfo.from_dict(v) for k, v in data['output_fields'].items()},
            '_instructions': data['instructions'],
            '__module__': '__main__',
        }
        return PromptMeta(data['name'], (Prompt,), namespace)