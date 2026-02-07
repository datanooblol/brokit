from brokit.primitives.prompt.core import Prompt
from brokit.primitives.prompt.types import Image, Audio
from brokit.primitives.lm.core import LM
from brokit.primitives.lm.types import ModelType, ModelResponse
from brokit.primitives.predictor.types import Prediction
from brokit.primitives.formatter import PromptFormatter
from brokit.primitives.shot import Shot
from brokit.primitives.history import PredictorHistory
from typing import Type, List, Optional
import time
import re

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

class Predictor:
    def __init__(self, prompt: Type[Prompt], lm:Optional[LM]=None, shots:Optional[list[Shot]]=None):
        self.prompt = prompt
        self.lm = lm
        self.shots = shots
        self.prompt_formatter = PromptFormatter()
        self.history: List[PredictorHistory] = []

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

    def _call_chat(self, lm, system_prompt, shot_prompt, input_prompt, images, audios):
        start = time.perf_counter()
        messages = self.prompt_formatter.format_chat(system_prompt, shot_prompt, input_prompt)
        model_response = lm(messages=messages, images=images, audios=audios)
        elapsed_ms = (time.perf_counter() - start) * 1000
        output = self.structure_output(model_response, self.prompt.output_fields)
        # response.parsed_response = output.to_dict()
        # response.request = messages
        # Create predictor history entry
        predictor_history = PredictorHistory(
            predictor_name=self.prompt.__name__,
            inputs=lm._serialize_request(messages),
            outputs=output.to_dict(),
            lm_call_id=lm.history[-1].id if lm.history else None,
            response_ms=elapsed_ms
        )
        self.history.append(predictor_history)        
        return output

    def __call__(self, images: Optional[List[Image]]=None, audios:Optional[List[Audio]]=None, **kwargs):
        input_fields = self.prompt.input_fields
        output_fields = self.prompt.output_fields
        instructions = self.prompt.instructions
        lm = self.lm
        system_prompt, input_prompt, shot_prompt = self.prompt_formatter(input_fields, output_fields, instructions, kwargs, self.shots)

        base64_images = None
        if images:
            base64_images = [
                img.to_base64() if isinstance(img, Image) else img 
                for img in images
            ]

        if lm.model_type == ModelType.CHAT:
            return self._call_chat(lm, system_prompt, input_prompt, shot_prompt, base64_images, audios)
        raise NotImplementedError("Only CHAT model type is implemented.")