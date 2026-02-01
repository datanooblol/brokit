from brokit.primitives.prompt import Prompt
from brokit.primitives.lm import LM, ModelType, ModelResponse
from typing import Type, List, Optional, Any
import re
from brokit.primitives.formatter import PromptFormatter
from brokit.primitives.prompt import Image
from brokit.primitives.shot import Shot

def parse_outputs(response: str, output_fields: dict, special_token: str = "<||{field}||>") -> dict:
    """Parse LM response with dynamic special tokens."""
    outputs = {}
    
    # Extract prefix and suffix from special_token template
    # e.g., "<||{field}||>" -> prefix="<||", suffix="||>"
    prefix, suffix = special_token.split("{field}")
    
    for field_name in output_fields.keys():
        # Escape special regex chars and build pattern
        escaped_prefix = re.escape(prefix)
        escaped_suffix = re.escape(suffix)
        pattern = rf"{escaped_prefix}{field_name}{escaped_suffix}\s*\n(.*?)(?={escaped_prefix}|$)"
        
        match = re.search(pattern, response, re.DOTALL)
        if match:
            outputs[field_name] = match.group(1).strip()
    
    return outputs

class Prediction:
    def __init__(self, **kwargs: Any) -> None:
        self._data = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self) -> str:
        items = ",\n    ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"Prediction(\n    {items}\n)"
    
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def to_dict(self) -> dict:
        return self._data        

class Predictor:
    def __init__(self, prompt: Type[Prompt], lm:Optional[LM]=None, shots:Optional[list[Shot]]=None):
        self.prompt = prompt
        self.lm = lm
        self.shots = shots
        self.prompt_formatter = PromptFormatter()

    def structure_output(self, response: ModelResponse, output_fields, special_token: str = "<||{field}||>") -> Prediction:
        output = parse_outputs(response.response, output_fields, special_token)
        
        # Convert types based on field definitions
        converted = {}
        for field_name, value in output.items():
            if field_name in output_fields:
                field_type = output_fields[field_name].type
                converted[field_name] = self._convert_type(value, field_type)
            else:
                converted[field_name] = value
        
        return Prediction(**converted)

    def _convert_type(self, value: str, target_type: type):
        """Convert string value to target type."""
        if target_type == str:
            return value
        elif target_type == int:
            return int(value.strip())
        elif target_type == float:
            return float(value.strip())
        elif target_type == bool:
            return value.strip().lower() in ('true', '1', 'yes')
        elif target_type == list:
            # Simple list parsing - can be enhanced
            return [item.strip() for item in value.strip('[]').split(',')]
        else:
            return value

    def _call_chat(self, lm, system_prompt, shot_prompt, input_prompt, images):
        messages = self.prompt_formatter.format_chat(system_prompt, shot_prompt, input_prompt, images)
        response = lm(messages=messages)
        output = self.structure_output(response, self.prompt.output_fields)
        response.parsed_response = output.to_dict()
        response.request = messages
        lm.history.append(response)
        return output

    def __call__(self, images: Optional[list[Image]]=None, **kwargs):
        # prompt_instance = self.prompt.to_dict()
        input_fields = self.prompt.input_fields
        output_fields = self.prompt.output_fields
        instructions = self.prompt.instructions
        lm = self.lm
        system_prompt, input_prompt, shot_prompt = self.prompt_formatter(input_fields, output_fields, instructions, kwargs, self.shots)

        base64_images = None
        if images:
            from brokit.primitives.prompt import Image
            base64_images = [
                img.to_base64() if isinstance(img, Image) else img 
                for img in images
            ]

        if lm.model_type == ModelType.CHAT:
            return self._call_chat(lm, system_prompt, input_prompt, shot_prompt, base64_images)
        raise NotImplementedError("Only CHAT model type is implemented.")