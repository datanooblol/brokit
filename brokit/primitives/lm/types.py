from typing import Dict, Optional, Any, Literal
from dataclasses import dataclass
from enum import Enum
from brokit.primitives.types import Field

class ModelType(str, Enum):
    CHAT = "chat"
    # COMPLETION = "completion" # not implement yet

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        return d

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> Dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}    

@dataclass
class ModelResponse:
    # Required fields (no default)
    model_name: str = Field(description="name of the model used")
    model_type: ModelType = Field(description="The type of the model (e.g., chat, completion)")
    response: str = Field(description="The raw response from the model")
    usage: Usage = Field(description="Token usage information")
    
    # Optional fields (with defaults)
    response_ms: Optional[float] = Field(description="Response time in milliseconds", default=None)
    cached: bool = Field(description="Whether the response was retrieved from cache", default=False)
    metadata: Optional[Dict[str, Any]] = Field(description="Additional metadata", default=None)
    request: Optional[Dict[str, Any]] = Field(description="All request data", default=None)
    parsed_response: Optional[Dict[str, Any]] = Field(description="Parsed response", default=None)
