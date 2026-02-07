from dataclasses import dataclass
from typing import Any, Optional, List, Dict
from datetime import datetime
import uuid
from brokit.primitives.types import Field

@dataclass
class BaseHistory:
    """Base class for all history entries"""
    id: str = Field(description="Unique call identifier", default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(description="When call was made", default_factory=datetime.now)
    error: Optional[str] = Field(description="Error if failed", default=None)

@dataclass
class LMHistory(BaseHistory):
    """Level 1: Raw LM interaction"""
    model_name: str = Field(description="Model name", default="")
    model_type: str = Field(description="Model type (chat/completion)", default="")
    request: Any = Field(description="Prompt or messages", default=None)
    response: str = Field(description="Raw model response", default="")
    usage: Optional[Dict[str, int]] = Field(description="Token usage", default=None)
    response_ms: float = Field(description="Response time in ms", default=0.0)
    cached: bool = Field(description="From cache", default=False)

@dataclass
class PredictorHistory(BaseHistory):
    """Level 2: Structured prediction"""
    predictor_name: str = Field(description="Prompt class name", default="")
    inputs: Dict[str, Any] = Field(description="Input field values", default_factory=dict)
    outputs: Dict[str, Any] = Field(description="Output field values", default_factory=dict)
    lm_call_id: Optional[str] = Field(description="Reference to LMCall", default=None)

@dataclass
class ProgramHistory(BaseHistory):
    """Level 3: Program execution step"""
    step_name: str = Field(description="Step/component name", default="")
    component_type: str = Field(description="Type: predictor/program/custom", default="")
    inputs: Dict[str, Any] = Field(description="Step inputs", default_factory=dict)
    outputs: Any = Field(description="Step outputs", default=None)
    child_call_ids: List[str] = Field(description="Child call references", default_factory=list)
