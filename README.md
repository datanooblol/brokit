# brokit

Inspired by big bro DSPy, brokit is a minimal Python toolkit of composable, LEGO-like primitives for working with language models. Build what you need, skip the bloat.

## What's This About?

A lightweight library for working with LMs across any use case. Just the essential building blocks, nothing more.

## Core Concepts

Coming from DSPy? You already know what's up:

- **Prompt** = `dspy.Signature` — Define your input/output structure
- **Predictor** = `dspy.Predict` — Execute prompts with your LM
- **LM** = `dspy.LM` — Language model interface
- **Shot** — Few-shot examples made simple

## Design Philosophy

### Plug and Play

Everything's a base class. Compose, extend, swap out whatever you want. The LM module? Bring your own.

### Pure Python

Zero required dependencies. Want to use `requests`, `httpx`, or `boto3`? Go for it. Check the notebooks for integration examples.

## Features

- Text and image support (more formats coming)
- Few-shot learning with Shot
- Build custom LM implementations
- Structured prompts with type hints

## Installation

```bash
pip install brokit
```

## Quick Start

```python
import brokit as bk

# Define your prompt structure
class QA(bk.Prompt):
    """Answer questions"""
    question: str = bk.InputField()
    answer: str = bk.OutputField()

# Create your LM and predictor
lm = YourLM(model_name="your-model")
qa = bk.Predictor(prompt=QA, lm=lm)

# Get results
response = qa(question="What is brokit?")
```

## Cookbook

Check out the `cookbook/` directory for hands-on examples:

**LM Integrations:**
- `lm/ollama.ipynb` - Ollama integration with vision support
- `lm/bedrock.ipynb` - AWS Bedrock integration

**Predictor Patterns:**
- `predictor/zero_shot.ipynb` - Basic prompting and chain-of-thought
- `predictor/few_shots.ipynb` - Few-shot learning with examples

## What's Next?

Check out [ROADMAP.md](ROADMAP.md) for what's coming and [VERSIONS.md](VERSIONS.md) for release notes.
