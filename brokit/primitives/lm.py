from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Literal
from dataclasses import dataclass
from enum import Enum
import time
from collections import OrderedDict
import json
import hashlib
from brokit.primitives.prompt import Image, Audio

class ModelType(str, Enum):
    CHAT = "chat"
    # COMPLETION = "completion" # not implement yet

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    # images: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        # if self.images:
        #     d["images"] = self.images
        return d

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int

@dataclass
class ModelResponse:
    model_name:str
    model_type:ModelType
    response:str
    usage:Usage
    response_ms: Optional[float] = None
    cached:bool = False
    metadata:Optional[Dict[str, Any]] = None
    request:Optional[Dict[str, Any]] = None
    parsed_response: Optional[Dict[str, Any]] = None

class LM(ABC):
    def __init__(self, model_name: str, model_type:ModelType, cache_size:int=10):
        self.model_name = model_name
        self.model_type = model_type
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self.history = []

    @abstractmethod    
    def request(self, prompt:Optional[str]=None, messages:Optional[List[Message]]=None, images:Optional[List[Image]]=None, audios:Optional[List[Audio]]=None,**kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, original_response:dict) -> ModelResponse:
        raise NotImplementedError

    def _validate_input(self, prompt:Optional[str], messages:Optional[List[Message]]):
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages must be provided")
        if prompt is not None and messages is not None:
            raise ValueError("Cannot provide both prompt and messages")

    def _cache_key(self, prompt: Optional[str], messages: Optional[List[Message]], images:Optional[List[Image]], audios:Optional[List[Audio]], kwargs: dict) -> str:
        """Generate cache key from request parameters."""
        # Convert messages to serializable format
        serializable_messages = None
        if messages:
            serializable_messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    # "images": msg.images # we must remove this and implement how to serialize images and audios
                } if isinstance(msg, Message) else msg
                for msg in messages
            ]
        
        cache_data = {
            "model": self.model_name,
            "prompt": prompt,
            "messages": serializable_messages,
            "images": [img._base64[:16] for img in images] if images else None, 
            **kwargs
        }
        # Convert to JSON string (sorted for consistency)
        json_str = json.dumps(cache_data, sort_keys=True)
        # Hash to fixed-length key
        return hashlib.sha256(json_str.encode()).hexdigest()    

    def __call__(self, prompt: Optional[str] = None, messages: Optional[List[Message]] = None, images:Optional[List[Image]]=None, audios:Optional[List[Audio]]=None, **kwargs) -> ModelResponse:
        self._validate_input(prompt, messages)
        key = self._cache_key(prompt, messages, images, audios, kwargs)
        # Check cache
        if key in self._cache:
            self._cache.move_to_end(key)  # Mark as recently used
            cached_response = self._cache[key]
            cached_response.cached = True
            return cached_response

        # Automatic timing
        start = time.perf_counter()
        original_response = self.request(prompt=prompt, messages=messages, images=images, audios=audios, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Parse response
        schema_response = self.parse_response(original_response)
        
        # Inject timing
        schema_response.response_ms = elapsed_ms
        # Add to cache with LRU eviction
        self._cache[key] = schema_response
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # Remove oldest        
        return schema_response