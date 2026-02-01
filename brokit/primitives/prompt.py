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
from typing import Type, Any, Dict, Union
import base64
from pathlib import Path
import httpx
import mimetypes

class Image:
    def __init__(self, source: Union[str, bytes, Path]):
        """
        Initialize Image from path, URL, or bytes.
        
        Args:
            source: File path, URL (http/https), or raw bytes
        """
        self.source = source
        if isinstance(source, bytes):
            self._base64 = base64.b64encode(source).decode('utf-8')
            self._mime_type = "image/jpeg"  # Default
        elif isinstance(source, (str, Path)):
            source_str = str(source)
            if source_str.startswith(('http://', 'https://')):
                self._base64 = self._from_url(source_str)
                self._mime_type = "image/jpeg"
            else:
                self._base64 = self._from_path(source_str)
                self._mime_type = mimetypes.guess_type(source_str)[0] or "image/jpeg"
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
    
    def _from_path(self, path: str) -> str:
        """Load image from file path."""
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _from_url(self, url: str) -> str:
        """Download image from URL."""
        response = httpx.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    
    def to_base64(self) -> str:
        """Get base64 encoded string."""
        return self._base64
    
    def __repr__(self) -> str:
        """DSPy-style representation."""
        data_url = f"data:{self._mime_type};base64,<IMAGE_BASE64_ENCODED({len(self._base64)})>"
        return f"Image(url={data_url})"

@dataclass
class FieldInfo:
    name: str
    description: str
    type: Type
    is_input: bool
    
def InputField(description:str = "") -> Any:
    return FieldInfo(name="", description=description, type=str, is_input=True)

def OutputField(description:str = "") -> Any:
    return FieldInfo(name="", description=description, type=str, is_input=False)

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