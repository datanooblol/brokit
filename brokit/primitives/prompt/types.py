from dataclasses import dataclass
from typing import Type, Any, Union
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

    def to_bytes(self) -> bytes:
        """Get raw bytes from base64."""
        return base64.b64decode(self._base64)        
    
    def __repr__(self) -> str:
        """DSPy-style representation."""
        data_url = f"data:{self._mime_type};base64,<IMAGE_BASE64_ENCODED({len(self._base64)})>"
        return f"Image(url={data_url})"

class Audio:
    def __init__(self, ): pass

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
