from dataclasses import field, MISSING

def Field(description=None, default=MISSING, default_factory=MISSING):
    """Helper to create dataclass fields with descriptions.
    
    Args:
        description: Human-readable description of the field
        default: Default value (use for immutable types)
        default_factory: Factory function for default (use for mutable types like list/dict)
    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Cannot specify both default and default_factory")
    
    input_fields = {}
    if description:
        input_fields["metadata"] = {"description": description}
    if default is not MISSING:
        input_fields["default"] = default
    if default_factory is not MISSING:
        input_fields["default_factory"] = default_factory
    
    return field(**input_fields)    
