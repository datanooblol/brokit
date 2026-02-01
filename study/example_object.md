# DSPy Example Object - Deep Dive

## Overview

The `Example` class is DSPy's flexible data container for training data and evaluation. It acts like a dictionary but with special features for machine learning workflows.

## Core Concept

**Example = Dictionary + ML Features**

- Store data with flexible schema (no predefined fields)
- Separate inputs (features) from labels (targets)
- Support both dot notation (`example.question`) and bracket notation (`example["question"]`)

## Architecture

### Internal Storage

```python
self._store = {}        # Holds all user data
self._input_keys = None # Tracks which fields are inputs
self._demos = []        # Stores demonstration examples
```

**Key insight:** User data lives in `_store`, not as real object attributes. This allows flexible schemas.

## Magic Methods Explained

### Attribute Access

```python
def __getattr__(self, key):
    # example.question → looks up _store["question"]
    if key in self._store:
        return self._store[key]
    raise AttributeError(...)

def __setattr__(self, key, value):
    # example.answer = "Paris" → stores in _store["answer"]
    if key.startswith("_") or key in dir(self.__class__):
        super().__setattr__(key, value)  # Real attributes
    else:
        self._store[key] = value  # User data
```

**Why?** Separates internal state from user data. Prevents name collisions.

### Dictionary Access

```python
def __getitem__(self, key):
    return self._store[key]  # example["question"]

def __setitem__(self, key, value):
    self._store[key] = value  # example["answer"] = "Paris"

def __contains__(self, key):
    return key in self._store  # "question" in example
```

**Why?** Provides familiar dict-like interface alongside attribute access.

### Hashing & Equality

```python
def __eq__(self, other):
    return isinstance(other, Example) and self._store == other._store

def __hash__(self):
    return hash(tuple(self._store.items()))
```

**Why hash?** 
- Enables deduplication: `set(examples)` removes duplicates
- Allows caching: `cache[example] = result`
- Required when defining `__eq__`

**Hashability in Python:**
- ✅ Hashable (immutable): `int`, `str`, `float`, `tuple`, `frozenset`
- ❌ Not hashable (mutable): `list`, `dict`, `set`

**The conversion:** `dict` → `dict.items()` → `tuple()` → hashable!

```python
_store = {"q": "Hi", "a": "Hello"}
hash(tuple(_store.items()))  # (("q", "Hi"), ("a", "Hello")) → hashable
```

## Input/Label Separation

The killer feature for ML workflows.

### with_inputs - Mark input fields

```python
def with_inputs(self, *keys):
    copied = self.copy()
    copied._input_keys = set(keys)
    return copied
```

**Usage:**
```python
example = Example(question="Hi", context="greeting", answer="Hello")
example = example.with_inputs("question", "context")
# Now: inputs = {question, context}, labels = {answer}
```

### inputs - Get only inputs

```python
def inputs(self):
    d = {key: self._store[key] for key in self._store if key in self._input_keys}
    new_instance = type(self)(base=d)
    new_instance._input_keys = self._input_keys
    return new_instance
```

**Returns:** New Example with only input fields.

### labels - Get only labels

```python
def labels(self):
    input_keys = self.inputs().keys()
    d = {key: self._store[key] for key in self._store if key not in input_keys}
    return type(self)(d)
```

**Returns:** New Example with everything except inputs.

**Visual:**
```
Example(question="Hi", context="greeting", answer="Hello")
         └──────────────┬──────────────┘  └─────┬─────┘
              with_inputs("question", "context")
                    ↓                           ↓
               inputs()                     labels()
```

## Advanced Patterns

### type(self) Pattern

```python
return type(self)(d)
```

**What it does:** Creates new instance of the current class (works with subclasses).

```python
# Instead of hardcoding:
return Example(d)  # ❌ Breaks inheritance

# Use:
return type(self)(d)  # ✅ Works with subclasses
```

**Example:**
```python
class CustomExample(Example):
    pass

custom = CustomExample(q="Hi")
result = custom.labels()  # Returns CustomExample, not Example!
```

### Serialization - toDict

```python
def toDict(self):
    def convert_to_serializable(value):
        if hasattr(value, "toDict"):
            return value.toDict()
        elif isinstance(value, BaseModel):  # Pydantic models
            return value.model_dump()
        elif isinstance(value, list):
            return [convert_to_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            return value
    
    return {k: convert_to_serializable(v) for k, v in self._store.items()}
```

**Why BaseModel check?** DSPy uses Pydantic models internally (like `dspy.History`). These need special handling to convert to plain dicts for JSON serialization.

**Conversion chain:**
1. Nested `Example` → call its `toDict()`
2. Pydantic model → `model_dump()`
3. List/dict → recursively convert
4. Primitives → use as-is

## Real-World Usage

### Training Data Preparation

```python
# Create training examples
examples = [
    Example(question="Capital of France?", answer="Paris").with_inputs("question"),
    Example(question="Capital of Japan?", answer="Tokyo").with_inputs("question"),
]

# During training
for example in examples:
    X = example.inputs()   # {"question": "..."}
    y = example.labels()   # {"answer": "..."}
    
    prediction = model(X)
    loss = compare(prediction, y)
```

### Deduplication

```python
# Remove duplicate examples
unique_examples = list(set(examples))  # Requires __hash__
```

### Caching

```python
# Cache expensive computations
cache = {}
if example not in cache:  # Requires __hash__ and __eq__
    cache[example] = expensive_computation(example)
```

## Key Takeaways

1. **Flexible storage:** `_store` dict holds all user data
2. **Dual interface:** Both `example.key` and `example["key"]` work
3. **ML-ready:** Separate inputs from labels for training
4. **Hashable:** Can use in sets/dicts for deduplication
5. **Inheritance-friendly:** `type(self)` pattern works with subclasses
6. **Serializable:** Handles complex nested objects including Pydantic models

## Python Concepts Used

- **Magic methods** (`__getattr__`, `__setitem__`, etc.) - Customize behavior
- **Dict comprehensions** - Filter and transform data
- **type()** - Get class of object for dynamic instantiation
- **Hashing** - Enable set/dict usage (requires immutability)
- **Property delegation** - Redirect attribute access to internal dict
