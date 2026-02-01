# DSPy Configuration Pattern Study

## Overview

This document explains how `dspy.configure()` works, when to use it, and design principles for building similar systems.

## How `dspy.configure()` Works

### Architecture

DSPy uses a **singleton-based global configuration** with thread-safe access and override capabilities.

**Key Components:**
- `Settings` class: Singleton managing configuration
- `main_thread_config`: Global configuration visible to all threads
- `thread_local_overrides`: Thread-specific overrides using `contextvars`
- `DEFAULT_CONFIG`: Initial default values

**Location in codebase:**
- Configuration: `dspy/dsp/utils/settings.py`
- Usage: `dspy/predict/predict.py` → `_forward_preprocess()`
- Execution: `dspy/adapters/base.py` → `__call__()`

### Execution Flow

```
1. User Code
   dspy.configure(lm=dspy.LM('openai/gpt-4'))
   ↓
2. Stores in settings.lm (global singleton)
   ↓
3. Module Call
   predictor = dspy.Predict("question -> answer")
   result = predictor(question="What is 2+2?")
   ↓
4. Retrieve LM (priority order)
   lm = kwargs.pop("lm", self.lm) or settings.lm
   ↓
5. Adapter formats and calls LM
   adapter(lm, lm_kwargs, signature, demos, inputs)
   ↓
6. Actual LM API call
   outputs = lm(messages=inputs, **lm_kwargs)
   ↓
7. Parse and return results
```

### Priority Order for LM Selection

1. **Call-level**: `predictor(question="...", lm=model)` (highest priority)
2. **Instance-level**: `predictor.lm = model`
3. **Global-level**: `dspy.configure(lm=model)` (fallback)

### Override Mechanisms

```python
# Global configuration
dspy.configure(lm=default_model)

# Instance override
predictor = dspy.Predict("q -> a", lm=instance_model)

# Call override
result = predictor(question="...", lm=call_model)

# Context override (temporary)
with dspy.context(lm=temp_model):
    result = predictor(question="...")
```

## Benefits of This Pattern

### 1. Convenience
Avoid repetitive LM passing across many modules.

```python
# Without global config
p1 = dspy.Predict("q -> a", lm=my_lm)
p2 = dspy.ChainOfThought("q -> a", lm=my_lm)
p3 = dspy.ReAct("q -> a", lm=my_lm)

# With global config
dspy.configure(lm=my_lm)
p1 = dspy.Predict("q -> a")
p2 = dspy.ChainOfThought("q -> a")
p3 = dspy.ReAct("q -> a")
```

### 2. Easy Model Switching
Change models globally in one place.

```python
# Development
dspy.configure(lm=dspy.LM('openai/gpt-3.5-turbo'))

# Production
dspy.configure(lm=dspy.LM('openai/gpt-4'))
```

### 3. Separation of Concerns
Decouple module logic from LM implementation.

```python
class MyModule(dspy.Module):
    def __init__(self):
        self.predictor = dspy.Predict("q -> a")  # No LM coupling
    
    def forward(self, question):
        return self.predictor(question=question)

# Configuration separate from logic
dspy.configure(lm=dspy.LM('openai/gpt-4'))
module = MyModule()
```

## Common Use Cases

### Use Case 1: Cost Optimization Research

**Scenario**: Test if cheaper models can achieve acceptable performance.

```python
# Optimize with powerful model to generate high-quality prompts
dspy.configure(lm=dspy.LM('openai/gpt-4'))
optimizer = dspy.BootstrapFewShot()
optimized_program = optimizer.compile(program, trainset=data)

# Test if cheaper model can use those optimized prompts
dspy.configure(lm=dspy.LM('openai/gpt-3.5-turbo'))
score = evaluate(optimized_program, testset=test_data)

# Decision: Deploy with cheaper model if score is acceptable
```

**When this makes sense:**
- ✅ Knowledge distillation (teacher-student pattern)
- ✅ Testing prompt transferability
- ✅ Cost-performance trade-off experiments

**When this doesn't make sense:**
- ❌ Final production evaluation (use deployment model)
- ❌ Benchmarking (use same model throughout)

### Use Case 2: A/B Testing

```python
# Test different models without code changes
if experiment_group == 'A':
    with dspy.context(lm=model_a):
        result = pipeline(input)
else:
    with dspy.context(lm=model_b):
        result = pipeline(input)
```

### Use Case 3: Multi-Stage Pipelines

**When global + context makes sense:**
```python
# Most modules use default, few need special treatment
dspy.configure(lm=default_model)

class Pipeline(dspy.Module):
    def __init__(self):
        self.step1 = dspy.Predict("q -> a")  # Uses default
        self.step2 = dspy.Predict("a -> b")  # Uses default
        self.step3 = dspy.Predict("b -> c")  # Uses default
        # Only critical step needs powerful model
        self.critical = dspy.Predict("c -> final", lm=powerful_model)
```

**When explicit is better:**
```python
# If you ALWAYS use different models, be explicit
class Pipeline(dspy.Module):
    def __init__(self):
        self.draft = dspy.Predict("q -> draft", lm=fast_model)
        self.refine = dspy.Predict("draft -> answer", lm=smart_model)
```

### Use Case 4: Development vs Production

```python
# Fast iteration in development
if os.getenv('ENV') == 'dev':
    dspy.configure(lm=dspy.LM('openai/gpt-3.5-turbo'))
else:
    dspy.configure(lm=dspy.LM('openai/gpt-4'))
```

**Important**: Use dev model only for:
- ✅ Rapid prototyping
- ✅ Testing code logic
- ❌ NOT for optimization or benchmarking

## Best Practices

### Golden Rule: Optimize and Deploy with Same Model

```python
# RECOMMENDED
model = dspy.LM('openai/gpt-4')

# Optimize
dspy.configure(lm=model)
optimized = optimizer.compile(program, trainset=train)

# Evaluate
score = evaluate(optimized, testset=test)

# Deploy (same model)
dspy.configure(lm=model)
result = optimized(input=user_input)
```

### When to Use Different Models

| Phase | Model | Reason |
|-------|-------|--------|
| Development | Cheap/Fast | Rapid iteration, code testing |
| Optimization | Target deployment model | Generate prompts for this model |
| Evaluation | Same as optimization | Accurate performance measurement |
| Production | Same as optimization | Consistency |
| Research | Various | Experimentation only |

## Design Patterns for Future Systems

### Pattern 1: Explicit Dependencies (No Global)

```python
class MySystem:
    def __init__(self, model):
        self.model = model
        self.predictor = dspy.Predict("q -> a", lm=model)
```

**Use when:**
- Small systems
- Clear model requirements
- Strict dependency tracking needed

**Pros:**
- Clear dependencies
- Easy to test
- No hidden state

**Cons:**
- Verbose
- Repetitive passing of dependencies

### Pattern 2: Global with Override (DSPy Pattern)

```python
dspy.configure(lm=default_model)
predictor = dspy.Predict("q -> a")  # Uses default
special = dspy.Predict("q -> a", lm=special_model)  # Override
```

**Use when:**
- Many modules
- Mostly same model
- Occasional exceptions needed

**Pros:**
- Convenient
- Flexible override
- Clean code

**Cons:**
- Hidden global state
- Harder to track dependencies
- Thread safety concerns

### Pattern 3: Dependency Injection (Enterprise)

```python
class ModelFactory:
    def create(self, env):
        if env == 'dev':
            return dspy.LM('openai/gpt-3.5-turbo')
        return dspy.LM('openai/gpt-4')

class MySystem:
    def __init__(self, model_factory):
        self.model = model_factory.create(os.getenv('ENV'))
```

**Use when:**
- Large systems
- Multiple environments
- Strict testing requirements
- Team collaboration

**Pros:**
- Full control
- Testable
- Environment-aware
- Clear contracts

**Cons:**
- More boilerplate
- Steeper learning curve

## Decision Matrix

| Scenario | Recommended Pattern | Reason |
|----------|-------------------|--------|
| Single model everywhere | Global config | Maximum convenience |
| Different models per module | Explicit `lm=` | Maximum clarity |
| Mostly same, few exceptions | Global + override | Balance convenience/clarity |
| A/B testing | Context manager | Temporary changes |
| Dev vs Prod | Environment-based config | Cost/speed in dev |
| Optimization research | Switch models | Experimentation |
| Production deployment | **Same as optimization** | Consistency |
| Large team project | Dependency injection | Testability |
| Prototype/Research | Global config | Speed of development |

## Key Takeaways

1. **Global configuration is a trade-off**: Convenience vs explicit dependencies
2. **Override hierarchy provides flexibility**: Call > Instance > Global > Default
3. **Context managers enable temporary changes**: Useful for testing and experimentation
4. **Optimize and deploy with the same model**: Only use different models for research
5. **Choose pattern based on system size and requirements**: No one-size-fits-all solution

## Technical Implementation Notes

### Thread Safety

- Only the "owner" thread (first to call `configure`) can modify global config
- Other threads see global config but cannot modify it
- Use `dspy.context()` for thread-local overrides
- Thread-local overrides propagate to child threads spawned with DSPy primitives

### Async Support

- Same thread ownership rules apply to async tasks
- IPython environments have special handling for interactive use
- Use `dspy.context()` in async tasks instead of `configure()`

### Configuration Storage

```python
DEFAULT_CONFIG = dotdict(
    lm=None,
    adapter=None,
    rm=None,
    trace=[],
    callbacks=[],
    async_max_workers=8,
    track_usage=False,
    max_history_size=10000,
    # ... more settings
)
```

All these settings follow the same pattern as `lm`.
