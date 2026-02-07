from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Callable
from brokit.primitives.history import ProgramHistory
import time

class Program(ABC):
    """Base class for multi-step workflows with automatic tracking"""
    history: List[ProgramHistory] = []
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Implement your workflow logic here"""
        raise NotImplementedError
    
    def _serialize_output(self, result: Any) -> Dict[str, Any]:
        """Convert output to dict for storage"""
        if isinstance(result, dict):
            return result
        elif hasattr(result, 'to_dict') and callable(result.to_dict):
            return result.to_dict()
        else:
            raise TypeError(
                f"Step output must be a dict or have a to_dict() method. "
                f"Got {type(result).__name__}. "
                f"Ensure your component returns a Prediction, dict, or object with to_dict()."
            )
    
    def step(self, step_name: str, component: Callable, **kwargs) -> Any:
        """Execute and track a component step"""
        error = None
        outputs = None
        result = None
        
        start = time.perf_counter()
        try:
            # Execute component
            result = component(**kwargs)
            
            # Serialize outputs (enforces contract)
            outputs = self._serialize_output(result)
            
        except Exception as e:
            error = str(e)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Collect child IDs from component history
            child_ids = []
            component_type = "custom"
            
            # Check if component has history (Predictor, LM, or Program)
            if hasattr(component, 'history') and component.history:
                component_type = type(component).__name__
                child_ids = [h.id for h in component.history]
            elif hasattr(component, '__self__'):  # Bound method fallback
                comp = component.__self__
                component_type = type(comp).__name__
                if hasattr(comp, 'history') and comp.history:
                    child_ids = [h.id for h in comp.history]
            
            # Track step in history
            step_history = ProgramHistory(
                step_name=step_name,
                component_type=component_type,
                inputs=kwargs,
                outputs=outputs,
                child_call_ids=child_ids,
                response_ms=elapsed_ms,
                error=error
            )
            self.history.append(step_history)
        
        return result
    
    def trace(self) -> str:
        """Get a simple text trace of execution"""
        lines = []
        for i, h in enumerate(self.history, 1):
            status = "✗" if h.error else "✓"
            lines.append(f"{i}. {status} {h.step_name} ({h.component_type})")
            if h.error:
                lines.append(f"   Error: {h.error}")
        return "\n".join(lines)
    
    def get_step(self, step_name: str) -> Optional[ProgramHistory]:
        """Get history for a specific step by name"""
        for h in self.history:
            if h.step_name == step_name:
                return h
        return None
    
    def __call__(self, **kwargs) -> Any:
        """Execute the program"""
        return self.forward(**kwargs)
