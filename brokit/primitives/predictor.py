from brokit.primitives.prompt import Prompt
from brokit.primitives.lm import LM, ModelType, ModelResponse
from typing import Type, List, Optional, Any
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

class PromptFormatter:
    def __call__(self, prompt_instance:dict, inputs:dict, shots:Optional[List[Prompt]]=None, special_token:str="<||{field}||>"):
        input_fields = prompt_instance.get('input_fields', {})
        output_fields = prompt_instance.get('output_fields', {})
        instructions = prompt_instance.get('instructions', '')
        system_prompt = []
        input_prompt = []
        self.format_system_in_out(system_prompt, input_fields, output_fields)
        self.format_system_structure(system_prompt, input_fields, output_fields, special_token)
        self.format_system_instruction(system_prompt, instructions)
        self.format_input_prompt(input_prompt, input_fields, output_fields, inputs, special_token)
        return system_prompt, input_prompt

    def _format_in_out(self, system_prompt:list, input_dict:dict)->list:
        idx = 1
        for field_name, field_value in input_dict.items():
            dtype = field_value.get('type', 'unknown')
            desc = field_value.get('description', '')
            system_prompt.append(f"{idx}. {field_name} ({dtype}): {desc}")
            idx += 1
        return system_prompt

    def format_system_in_out(self, system_prompt, input_fields, output_fields)->list:
        system_prompt.append("Your input fields are:")
        self._format_in_out(system_prompt, input_fields) 
        system_prompt.append("Your output fields are:")
        self._format_in_out(system_prompt, output_fields) 
        return system_prompt

    def _format_structure(self, system_prompt:list, input_dict:dict, special_token:str)->list:
        for field_name, field_value in input_dict.items():
            system_prompt.append(f"{special_token.format(field=field_name)}\n{{{field_name}}}\n")
        return system_prompt

    def format_system_structure(self, system_prompt:list, input_fields:dict, output_fields:dict, special_token:str)->list:
        system_prompt.append("\nAll interactions will be structured in the following way, with the appropriate values filled in.\n")
        self._format_structure(system_prompt, input_fields, special_token)
        self._format_structure(system_prompt, output_fields, special_token)
        system_prompt.append(special_token.format(field="completed"))
        return system_prompt
    
    def format_system_instruction(self, system_prompt:list, instructions:str)->list:
        system_prompt.append("In adhering to this structure, your objective is: ")
        system_prompt.append(instructions)
        return system_prompt
    
    def format_input_prompt(self, input_prompt:list, input_fields:dict, output_fields:dict, inputs:dict, special_token:str)->list:
        for input_name, input_value in inputs.items():
            if input_name in input_fields:
                input_prompt.append(f"{special_token.format(field=input_name)}\n{input_value}\n")
        input_prompt.append(
            "Respond with the corresponding output fields, starting with the field, " +
            ''.join([f"`{special_token.format(field=field_name)}`" for field_name in output_fields.keys()]) +
            f" and then ending with the marker for `{special_token.format(field='completed')}`."
        )
        return input_prompt

    @staticmethod
    def format_chat(system_prompt:list, input_prompt:list)->list:
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": "\n".join(system_prompt)
            })
        if input_prompt:
            messages.append({
                "role": "user",
                "content": "\n".join(input_prompt)
            })
        return messages
    
    @staticmethod
    def parse_prediction(response:str, output_fields:dict, special_token:str="<||{field}||>")->dict:
        outputs = {}
        for field_name in output_fields.keys():
            start_token = special_token.format(field=field_name)
            end_token = special_token.format(field="completed")
            start_idx = response.find(start_token)
            if start_idx != -1:
                start_idx += len(start_token)
                end_idx = response.find(end_token, start_idx)
                if end_idx == -1:
                    end_idx = len(response)
                field_value = response[start_idx:end_idx].strip()
                outputs[field_name] = field_value
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
    def __init__(self, prompt: Type[Prompt], lm:Optional[LM]=None, shots:Optional[List[Prompt]]=None):
        self.prompt = prompt
        self.lm = lm
        self.shots = shots
        self.prompt_formatter = PromptFormatter()

    def structure_output(self, response:ModelResponse, output_fields, special_token:str="<||{field}||>")->Prediction:
        output = parse_outputs(response.response, output_fields, special_token)
        return Prediction(**output)

    def _call_chat(self, lm, system_prompt, input_prompt):
        messages = self.prompt_formatter.format_chat(system_prompt, input_prompt)
        response = lm(messages=messages)
        output = self.structure_output(response, self.prompt.output_fields)
        response.parsed_response = output.to_dict()
        lm.history.append(response)
        return output

    def __call__(self, **kwargs):
        prompt_instance = self.prompt.to_dict()
        lm = self.lm
        system_prompt, input_prompt = self.prompt_formatter(prompt_instance, kwargs, self.shots)
        if lm.model_type == ModelType.CHAT:
            return self._call_chat(lm, system_prompt, input_prompt)
        raise NotImplementedError("Only CHAT model type is implemented.")