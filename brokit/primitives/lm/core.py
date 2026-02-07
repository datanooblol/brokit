from abc import ABC, abstractmethod
from typing import List, Optional, Any
import time
from collections import OrderedDict
import json
import hashlib
from brokit.primitives.prompt.types import Image, Audio
from brokit.primitives.lm.types import ModelType, Message, ModelResponse
from brokit.primitives.history import LMHistory

class LM(ABC):
    def __init__(self, model_name: str, model_type:ModelType, cache_size:int=10):
        self.model_name = model_name
        self.model_type = model_type
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self.history: List[LMHistory] = []

    @abstractmethod    
    def request(self, prompt:Optional[str]=None, messages:Optional[List[Message]]=None, images:Optional[List[Image]]=None, audios:Optional[List[Audio]]=None,**kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def response(self, original_response:dict) -> ModelResponse:
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

    def _serialize_request(self, request: Any) -> Any:
        """Convert request to serializable format"""
        if isinstance(request, str):
            return request
        if isinstance(request, list):
            return [msg.to_dict() if isinstance(msg, Message) else msg for msg in request]
        return request

    def _track_call(self, model_response: ModelResponse, request: Any) -> None:
        """Track call in history"""
        lm_call = LMHistory(
            model_name=self.model_name,
            model_type=self.model_type.value,
            request=self._serialize_request(request),
            response=model_response.response,
            usage=model_response.usage.to_dict(),
            response_ms=model_response.response_ms or 0.0,
            cached=model_response.cached
        )
        self.history.append(lm_call)

    def __call__(self, prompt: Optional[str] = None, messages: Optional[List[Message]] = None, 
                images: Optional[List[Image]] = None, audios: Optional[List[Audio]] = None, **kwargs) -> ModelResponse:
        self._validate_input(prompt, messages)
        key = self._cache_key(prompt, messages, images, audios, kwargs)
        
        # Check cache
        if key in self._cache:
            self._cache.move_to_end(key)
            cached_response = self._cache[key]
            cached_response.cached = True
            self._track_call(cached_response, messages or prompt)
            return cached_response

        # Automatic timing
        start = time.perf_counter()
        original_response = self.request(prompt=prompt, messages=messages, images=images, audios=audios, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Parse response
        model_response = self.response(original_response)
        model_response.response_ms = elapsed_ms
        
        # Track and cache
        self._track_call(model_response, messages or prompt)
        self._cache[key] = model_response
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        
        return model_response    

    # def __call__(self, prompt: Optional[str] = None, messages: Optional[List[Message]] = None, 
    #             images: Optional[List[Image]] = None, audios: Optional[List[Audio]] = None, **kwargs) -> ModelResponse:
    #     self._validate_input(prompt, messages)
    #     key = self._cache_key(prompt, messages, images, audios, kwargs)
        
    #     # Check cache
    #     if key in self._cache:
    #         self._cache.move_to_end(key)
    #         cached_response = self._cache[key]
    #         cached_response.cached = True
    #         return cached_response

    #     # Automatic timing
    #     start = time.perf_counter()
    #     original_response = self.request(prompt=prompt, messages=messages, images=images, audios=audios, **kwargs)
    #     elapsed_ms = (time.perf_counter() - start) * 1000
        
    #     # Parse response
    #     model_response = self.response(original_response)
    #     model_response.response_ms = elapsed_ms
        
    #     # Create history entry
    #     lm_call = LMHistory(
    #         model_name=self.model_name,
    #         model_type=self.model_type.value,
    #         request=self._serialize_request(messages or prompt),
    #         response=model_response.response,
    #         usage=model_response.usage.to_dict(),
    #         response_ms=elapsed_ms,
    #         cached=False
    #     )
    #     self.history.append(lm_call)
        
    #     # Add to cache with LRU eviction
    #     self._cache[key] = model_response
    #     if len(self._cache) > self._cache_size:
    #         self._cache.popitem(last=False)
        
    #     return model_response
