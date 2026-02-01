from brokit.primitives.prompt import Prompt
from brokit.primitives.shot import Shot
from brokit.primitives.lm import Message
from typing import List, Optional

class PromptFormatter:
    def __call__(
            self, 
            input_fields:dict,
            output_fields:dict,
            instructions:str,
            inputs:dict, 
            shots:Optional[List[Shot]]=None, 
            special_token:str="<||{field}||>"
        ):
        system_prompt = []
        input_prompt = []
        shot_prompt = []
        self.format_system_in_out(system_prompt, input_fields, output_fields)
        self.format_system_structure(system_prompt, input_fields, output_fields, special_token)
        self.format_system_instruction(system_prompt, instructions)
        self.format_shot_prompt(shot_prompt, input_fields, output_fields, shots, special_token)
        self.format_input_prompt(input_prompt, input_fields, output_fields, inputs, special_token)
        return system_prompt, shot_prompt, input_prompt

    def format_shot_prompt(self, shot_prompt: list, input_fields: dict, output_fields: dict, shots: Optional[List], special_token: str) -> list:
        """Format shots as user/assistant message pairs."""
        from brokit.primitives.lm import Message
        
        if not shots:
            return shot_prompt
        
        for shot in shots:
            # User message with inputs
            user_content = []
            for field_name in input_fields.keys():
                value = shot.inputs.get(field_name, "")
                user_content.append(f"{special_token.format(field=field_name)}\n{value}")
            
            shot_prompt.append(Message(
                role="user",
                content="\n".join(user_content)
            ))
            
            # Assistant message with outputs
            assistant_content = []
            for field_name in output_fields.keys():
                value = shot.outputs.get(field_name, "Intentionally left blank.")
                assistant_content.append(f"{special_token.format(field=field_name)}\n{value}\n")
            assistant_content.append(special_token.format(field="completed"))
            
            shot_prompt.append(Message(
                role="assistant",
                content="\n".join(assistant_content)
            ))
        
        return shot_prompt

    def _format_in_out(self, system_prompt:list, input_dict:dict)->list:
        idx = 1
        for field_name, field_value in input_dict.items():
            dtype = field_value.type
            desc = field_value.description
            system_prompt.append(f"{idx}. {field_name} ({dtype}): {desc}")
            idx += 1
        return system_prompt

    def format_system_in_out(self, system_prompt, input_fields, output_fields)->list:
        if input_fields:
            system_prompt.append("Your input fields are:")
            self._format_in_out(system_prompt, input_fields) 
        if output_fields:
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
    
    def format_input_prompt(self, input_prompt: list, input_fields: dict, output_fields: dict, inputs: dict, special_token: str) -> list:
        for input_name, input_value in inputs.items():
            if input_name in input_fields:
                input_prompt.append(f"{special_token.format(field=input_name)}\n{input_value}\n")
        
        input_prompt.append(
            "Respond with the corresponding output fields, starting with the field: " +
            ', '.join([f"`{special_token.format(field=field_name)}`" for field_name in output_fields.keys()]) +
            f" and then ending with the marker for `{special_token.format(field='completed')}`."
        )
        
        return input_prompt

    # @staticmethod
    # def format_chat(system_prompt: list, input_prompt: list, images: Optional[list] = None) -> list[Message]:
    #     messages = []
    #     if system_prompt:
    #         messages.append(Message(role="system", content="\n".join(system_prompt)))
    #     if input_prompt:
    #         messages.append(Message(
    #             role="user", 
    #             content="\n".join(input_prompt),
    #             images=images
    #         ))
    #     return messages   
    @staticmethod
    def format_chat(system_prompt: list, shot_prompt: list, input_prompt: list, images: Optional[list] = None) -> list[Message]:
        """
        Format messages in order: system → shots → user input.
        
        Args:
            system_prompt: System instructions
            shot_prompt: List of Message objects (user/assistant pairs)
            input_prompt: User input
            images: Optional images for user input
        """
        from brokit.primitives.lm import Message
        
        messages = []
        
        # 1. System message
        if system_prompt:
            messages.append(Message(role="system", content="\n".join(system_prompt)))
        
        # 2. Shot messages (already Message objects)
        if shot_prompt:
            messages.extend(shot_prompt)
        
        # 3. User input message
        if input_prompt:
            messages.append(Message(
                role="user",
                content="\n".join(input_prompt),
                images=images
            ))
        
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
