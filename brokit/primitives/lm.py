from typing import Optional, Any, List, Dict, Callable, Literal
from dataclasses import dataclass

@dataclass
class Usage:
    input_tokens:int = 0
    output_tokens:int = 0

@dataclass
class ModelResponse:
    model_name:str
    model_type:Literal["chat", "completion"]
    content:str
    usage:Usage

class BaseLM:
    def __init__(self, request_fn:Callable, response_fn:Callable):
        self.request_fn = request_fn
        self.response_fn = response_fn

    def __call__(self, prompt:Optional[str]=None, messages:Optional[List[Dict[str, Any]]]=None, **kwargs) -> ModelResponse:
        return self.forward(prompt=prompt, messages=messages)
    
    def forward(self, prompt:Optional[str]=None, messages:Optional[List[Dict[str, Any]]]=None, **kwargs) -> ModelResponse:
        input_params = dict(
            prompt=prompt,
            messages=messages
        )
        response = self.request_fn(**input_params)
        return self.response_fn(response)

class LM(BaseLM):
    pass