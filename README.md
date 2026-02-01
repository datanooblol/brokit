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

## Getting Started

Peep the notebooks:
- Custom Prompt signatures
- Your own LM implementations  
- Few-shot examples
- External library integrations

## What's Next?

Check out [ROADMAP.md](ROADMAP.md) for what's coming and [VERSIONS.md](VERSIONS.md) for release notes.
