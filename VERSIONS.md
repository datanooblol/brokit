# Versions

## 0.1.2 - History & Program

**What's new:**
- **History tracking** - Complete execution traces for debugging
  - `LMHistory` - Track raw model calls (request, response, tokens, timing, cache hits)
  - `PredictorHistory` - Track structured predictions with links to LM calls
  - `ProgramHistory` - Track multi-step workflows with links to child calls
- **Program primitive** - `bk.Program` for composing multi-step workflows
  - Simple `step()` method to execute and track components
  - Auto-captures inputs, outputs, timing, and errors
  - Works with Predictors, LMs, custom functions, or nested Programs
  - Debug helpers: `trace()` for quick overview, `get_step()` for inspection
- **Unified timing** - All history entries track execution time in milliseconds
- **Serializable history** - All history uses base Python types (dict, str, int) for easy export/analysis

## 0.1.1 - Vision & Imports

**What's new:**
- **Import style** - Use `import brokit as bk` (DSPy-style imports)
- **Vision support** - `bk.Image` class for working with vision models
  - Load from file path, URL, or bytes
  - Integrated examples in `cookbook/lm/ollama.ipynb` and `cookbook/lm/bedrock.ipynb`
- **Cookbook** - Added practical notebooks for LM integrations and predictor patterns

## 0.1.0 - The Foundation Drop

First release, let's go! 🚀

**What's in the box:**
- **Prompt** - Define your input/output structure like a boss
- **LM** - Plug in any language model you want
- **Predictor** - Run your prompts and get results
- **Shot** - Few-shot examples made easy
- **Image support** - Predictor works with images (more formats coming soon)

This is the core toolkit. Everything you need to start building with LMs, nothing you don't.
