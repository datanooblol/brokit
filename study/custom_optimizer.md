# Building Custom DSPy Optimizers

## What is an Optimizer?

An **optimizer** transforms a zero-shot DSPy module into a few-shot module by selecting good examples (demos) and/or improving instructions.

```python
# Before optimization
module = SimpleQA()
result = module(question="What is Python?")  # Zero-shot

# After optimization
optimizer = dspy.BootstrapFewShot(metric=metric)
compiled = optimizer.compile(module, trainset=trainset)
result = compiled(question="What is Python?")  # Few-shot with demos
```

---

## The Standard Interface

All DSPy optimizers follow the **Strategy Pattern** with a uniform interface:

```python
class Optimizer:
    def __init__(self, metric, **config):
        self.metric = metric
        self.config = config
    
    def compile(self, student, trainset, valset=None, **kwargs):
        # Returns optimized copy of student
        pass
```

**Key insight:** Same interface allows swapping optimizers without changing code.

---

## What Optimizers Modify

### 1. predictor.demos (Few-Shot Examples)
```python
for name, predictor in student.named_predictors():
    predictor.demos = [demo1, demo2, demo3]
```

### 2. predictor.signature.instructions (System Prompt)
```python
for name, predictor in student.named_predictors():
    predictor.signature.instructions = "Be concise and accurate."
```

### 3. Other Attributes
```python
predictor.temperature = 0.7
predictor.max_tokens = 500
```

---

## Essential Rules

### ✅ DO

1. **Always deepcopy() the student**
   ```python
   def compile(self, student, trainset):
       optimized = student.deepcopy()  # Don't modify original!
       # ... optimize ...
       return optimized
   ```

2. **Use named_predictors() to find predictors**
   ```python
   for name, predictor in optimized.named_predictors():
       predictor.demos = selected_demos
   ```

3. **Mark as compiled**
   ```python
   optimized._compiled = True
   ```

4. **Handle errors gracefully**
   ```python
   for example in trainset:
       try:
           prediction = teacher(**example.inputs())
           if self.metric(example, prediction):
               demos.append(example)
       except Exception as e:
           print(f"Skipping: {e}")
           continue
   ```

5. **Validate with metric**
   ```python
   if self.metric(example, prediction):
       demos.append(example)
   ```

### ❌ DON'T

1. **Don't modify original student**
   ```python
   # ❌ Bad
   def compile(self, student, trainset):
       for name, pred in student.named_predictors():
           pred.demos = demos
       return student
   ```

2. **Don't forget to mark as compiled**
   ```python
   # ❌ Missing
   return optimized
   
   # ✅ Correct
   optimized._compiled = True
   return optimized
   ```

3. **Don't assume predictor structure**
   ```python
   # ❌ Assumes single predictor
   student.predictor.demos = demos
   
   # ✅ Works for any structure
   for name, pred in student.named_predictors():
       pred.demos = demos
   ```

---

## Optimization Strategies

### 1. Random Selection
```python
import random

class RandomOptimizer:
    def __init__(self, metric, num_demos=3):
        self.metric = metric
        self.num_demos = num_demos
    
    def compile(self, student, trainset, valset=None):
        optimized = student.deepcopy()
        demos = random.sample(trainset, min(self.num_demos, len(trainset)))
        
        for name, predictor in optimized.named_predictors():
            predictor.demos = demos
        
        optimized._compiled = True
        return optimized
```

### 2. Similarity-Based
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityOptimizer:
    def __init__(self, metric, num_demos=3):
        self.metric = metric
        self.num_demos = num_demos
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def compile(self, student, trainset, valset):
        optimized = student.deepcopy()
        
        # Encode examples
        train_texts = [self._to_text(ex) for ex in trainset]
        val_texts = [self._to_text(ex) for ex in valset]
        
        train_emb = self.encoder.encode(train_texts)
        val_emb = self.encoder.encode(val_texts)
        
        # Find most representative
        similarities = cosine_similarity(train_emb, val_emb)
        avg_sim = similarities.mean(axis=1)
        top_indices = avg_sim.argsort()[-self.num_demos:]
        
        demos = [trainset[i] for i in top_indices]
        
        for name, predictor in optimized.named_predictors():
            predictor.demos = demos
        
        optimized._compiled = True
        return optimized
    
    def _to_text(self, example):
        return " ".join(str(v) for v in example.values())
```

### 3. Validation-Based (Grid Search)
```python
class ValidationOptimizer:
    def __init__(self, metric, num_demos=3, num_trials=20):
        self.metric = metric
        self.num_demos = num_demos
        self.num_trials = num_trials
    
    def compile(self, student, trainset, valset):
        optimized = student.deepcopy()
        
        best_score = 0
        best_demos = None
        
        # Try multiple random demo sets
        for _ in range(self.num_trials):
            demo_set = random.sample(trainset, self.num_demos)
            
            # Apply and evaluate
            for name, predictor in optimized.named_predictors():
                predictor.demos = demo_set
            
            score = self._evaluate(optimized, valset)
            
            if score > best_score:
                best_score = score
                best_demos = demo_set
        
        # Apply best
        for name, predictor in optimized.named_predictors():
            predictor.demos = best_demos
        
        optimized._compiled = True
        return optimized
    
    def _evaluate(self, student, valset):
        correct = 0
        for example in valset:
            try:
                pred = student(**example.inputs())
                if self.metric(example, pred):
                    correct += 1
            except:
                pass
        return correct / len(valset) if valset else 0
```

### 4. Teacher-Student (Like BootstrapFewShot)
```python
class TeacherStudentOptimizer:
    def __init__(self, metric, teacher_lm, num_demos=3):
        self.metric = metric
        self.teacher_lm = teacher_lm
        self.num_demos = num_demos
    
    def compile(self, student, trainset, valset=None):
        # Create teacher with stronger model
        teacher = student.deepcopy()
        teacher.set_lm(self.teacher_lm)
        
        # Generate demos
        demos = []
        for example in trainset:
            try:
                # Teacher generates output
                prediction = teacher(**example.inputs())
                
                # Validate
                if self.metric(example, prediction):
                    # Merge input + output
                    demo = example.copy()
                    for key, value in prediction.items():
                        demo[key] = value
                    demos.append(demo)
                
                if len(demos) >= self.num_demos:
                    break
            except:
                continue
        
        # Apply to student
        optimized = student.deepcopy()
        for name, predictor in optimized.named_predictors():
            predictor.demos = demos
        
        optimized._compiled = True
        return optimized
```

### 5. Diversity-Based
```python
from collections import defaultdict

class DiversityOptimizer:
    def __init__(self, metric, num_demos=5):
        self.metric = metric
        self.num_demos = num_demos
    
    def compile(self, student, trainset, valset=None):
        optimized = student.deepcopy()
        
        # Cluster by output length
        clusters = self._cluster(trainset)
        
        # Select from each cluster
        demos = []
        for cluster in clusters:
            if len(demos) >= self.num_demos:
                break
            demos.append(random.choice(cluster))
        
        # Fill remaining
        while len(demos) < self.num_demos and len(demos) < len(trainset):
            candidate = random.choice(trainset)
            if candidate not in demos:
                demos.append(candidate)
        
        for name, predictor in optimized.named_predictors():
            predictor.demos = demos
        
        optimized._compiled = True
        return optimized
    
    def _cluster(self, trainset):
        clusters = defaultdict(list)
        for ex in trainset:
            output_key = list(ex.labels().keys())[0]
            output_len = len(str(ex[output_key]))
            bucket = output_len // 50
            clusters[bucket].append(ex)
        return list(clusters.values())
```

---

## Complete Template

```python
import dspy
import random

class MyCustomOptimizer:
    """Template for custom optimizer"""
    
    def __init__(self, metric, **config):
        """
        Args:
            metric: Validation function (example, prediction) -> bool
            **config: Your custom configuration
        """
        self.metric = metric
        self.config = config
    
    def compile(self, student, trainset, valset=None, **kwargs):
        """
        Args:
            student: dspy.Module to optimize
            trainset: List of dspy.Example with .with_inputs()
            valset: Optional validation set
        
        Returns:
            Optimized copy of student
        """
        # 1. Copy student (don't modify original)
        optimized = student.deepcopy()
        
        # 2. Find predictors
        predictors = list(optimized.named_predictors())
        
        # 3. Your optimization logic
        demos = self._select_demos(trainset, valset)
        
        # 4. Apply to all predictors
        for name, predictor in predictors:
            predictor.demos = demos
            # Optionally modify instructions
            # predictor.signature.instructions = "..."
        
        # 5. Mark as compiled
        optimized._compiled = True
        
        return optimized
    
    def _select_demos(self, trainset, valset):
        """Your demo selection logic"""
        # Example: random selection
        num_demos = self.config.get('num_demos', 3)
        return random.sample(trainset, min(num_demos, len(trainset)))
    
    def _evaluate(self, student, dataset):
        """Helper to evaluate student on dataset"""
        correct = 0
        for example in dataset:
            try:
                prediction = student(**example.inputs())
                if self.metric(example, prediction):
                    correct += 1
            except Exception as e:
                print(f"Error: {e}")
                continue
        return correct / len(dataset) if dataset else 0
```

---

## Usage Pattern

```python
import dspy

# 1. Create module
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")
    
    def forward(self, question):
        return self.predictor(question=question)

# 2. Prepare data
trainset = [
    dspy.Example(question="Q1", answer="A1").with_inputs("question"),
    dspy.Example(question="Q2", answer="A2").with_inputs("question"),
]

# 3. Define metric
def validate(example, prediction, trace=None):
    return example.answer.lower() in prediction.answer.lower()

# 4. Create optimizer
optimizer = MyCustomOptimizer(metric=validate, num_demos=3)

# 5. Compile
student = SimpleQA()
compiled = optimizer.compile(student, trainset=trainset)

# 6. Use compiled version
result = compiled(question="What is Python?")

# 7. Save
compiled.save("compiled_model.json")
```

---

## Key Takeaways

| Concept | Lesson |
|---------|--------|
| **Interface** | Use `.compile(student, trainset, valset)` |
| **Immutability** | Always `deepcopy()` student |
| **Discovery** | Use `named_predictors()` to find predictors |
| **Modification** | Set `predictor.demos` and/or `predictor.signature.instructions` |
| **Validation** | Use `metric(example, prediction)` to validate |
| **Error Handling** | Wrap in try-except, skip bad examples |
| **Marking** | Set `_compiled = True` |
| **Strategy** | Random, similarity, validation, teacher-student, diversity |

---

## Advanced Topics

### Per-Predictor Demos
```python
# Different demos for different predictors
for name, predictor in optimized.named_predictors():
    if "classifier" in name:
        predictor.demos = classification_demos
    elif "generator" in name:
        predictor.demos = generation_demos
```

### Instruction Optimization
```python
# Optimize instructions instead of demos
for name, predictor in optimized.named_predictors():
    predictor.signature.instructions = optimized_instruction
```

### Hybrid Optimization
```python
# Optimize both demos and instructions
for name, predictor in optimized.named_predictors():
    predictor.demos = selected_demos
    predictor.signature.instructions = optimized_instruction
```

---

## Multi-Predictor Programs: End-to-End vs Modular Optimization

### The Challenge

When you have a program with multiple predictors:

```python
class ThreeStepProgram(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.Predict("question -> thought")
        self.predict2 = dspy.Predict("thought -> analysis")
        self.predict3 = dspy.Predict("analysis -> answer")
    
    def forward(self, question):
        step1 = self.predict1(question=question)
        step2 = self.predict2(thought=step1.thought)
        step3 = self.predict3(analysis=step2.analysis)
        return step3
```

You face a choice: **optimize together** or **optimize separately**?

### How Bootstrap Creates Per-Predictor Demos

You only provide end-to-end examples:
```python
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question")
]
```

Bootstrap **traces execution** and captures intermediate steps:

```python
# During optimization, for "Capital of France?" example:
# 1. Execute program with tracing
with dspy.context(trace=[]):
    result = teacher(question="Capital of France?")
    trace = dspy.settings.trace

# 2. Trace captures each predictor call:
# trace = [
#   (predict1, {"question": "Capital of France?"}, {"thought": "Paris"}),
#   (predict2, {"thought": "Paris"}, {"analysis": "City of Light..."}),
#   (predict3, {"analysis": "City of Light..."}, {"answer": "Paris"})
# ]

# 3. If metric passes, create demos from trace:
for predictor, inputs, outputs in trace:
    demo = dspy.Example(augmented=True, **inputs, **outputs)
    predictor_demos[predictor_name].append(demo)

# Result: Each predictor gets its own demos!
# predict1.demos = [Example(question="Capital of France?", thought="Paris")]
# predict2.demos = [Example(thought="Paris", analysis="City of Light...")]
# predict3.demos = [Example(analysis="City of Light...", answer="Paris")]
```

### Two Types of Demos

```python
# After optimization, each predictor has:
predictor.demos = augmented_demos + raw_demos

# augmented_demos: From successful traces (intermediate steps)
# Example({'augmented': True, 'question': '...', 'thought': '...'})

# raw_demos: Original examples that failed bootstrapping
# Example({'question': '...', 'answer': '...'})
```

### Strategy Comparison

#### End-to-End Optimization (Default)

**How it works:**
```python
optimizer = dspy.BootstrapFewShot(metric=validate_answer)
compiled = optimizer.compile(program, trainset=trainset)
# All predictors optimized together via tracing
```

**Pros:**
- ✅ Time efficient - One optimization run
- ✅ Coherent pipeline - Steps learn to work together
- ✅ Realistic traces - Demos reflect actual execution
- ✅ Less manual work - No intermediate labels needed

**Cons:**
- ❌ Error propagation - Bad step1 → bad step2 → bad step3
- ❌ Hard to debug - Which predictor is failing?
- ❌ Quality uncertainty - Intermediate outputs might be nonsense
- ❌ All-or-nothing - If one step fails, whole trace is lost

#### Modular Optimization (Manual)

**How it works:**
```python
# Optimize each predictor separately
step1_data = [dspy.Example(question="...", thought="...").with_inputs("question")]
step2_data = [dspy.Example(thought="...", analysis="...").with_inputs("thought")]
step3_data = [dspy.Example(analysis="...", answer="...").with_inputs("analysis")]

opt1 = dspy.BootstrapFewShot(metric=validate_thought)
program.predict1 = opt1.compile(program.predict1, trainset=step1_data)

opt2 = dspy.BootstrapFewShot(metric=validate_analysis)
program.predict2 = opt2.compile(program.predict2, trainset=step2_data)

opt3 = dspy.BootstrapFewShot(metric=validate_answer)
program.predict3 = opt3.compile(program.predict3, trainset=step3_data)
```

**Pros:**
- ✅ Isolated debugging - Test each step independently
- ✅ Quality control - Verify intermediate outputs make sense
- ✅ Targeted improvement - Fix specific weak steps
- ✅ Reusable components - Use step1 in different programs

**Cons:**
- ❌ More time - 3 separate optimization runs
- ❌ Manual labeling - Need to create intermediate labels
- ❌ Distribution mismatch - Training data ≠ real execution
- ❌ No end-to-end coherence - Steps might not work well together

### Recommended Hybrid Approach

```python
# 1. Start with end-to-end optimization
optimizer = dspy.BootstrapFewShot(metric=validate_answer)
compiled = optimizer.compile(program, trainset=trainset)

# 2. Inspect each predictor's demos
for name, predictor in compiled.named_predictors():
    print(f"{name}: {len(predictor.demos)} demos")
    for demo in predictor.demos[:3]:
        print(demo)

# 3. If a specific predictor is weak, optimize it separately
if step2_quality_is_low:
    # Create targeted dataset for step2
    step2_data = create_step2_examples()
    step2_optimizer = dspy.BootstrapFewShot(metric=validate_step2)
    compiled.predict2 = step2_optimizer.compile(
        compiled.predict2, 
        trainset=step2_data
    )

# 4. Re-optimize end-to-end to ensure coherence
final_optimizer = dspy.BootstrapFewShot(metric=validate_answer)
final_compiled = final_optimizer.compile(compiled, trainset=trainset)
```

### When to Use Each Strategy

**Use End-to-End When:**
- Building a new system quickly
- You have good end-to-end metrics
- Steps are tightly coupled
- You lack intermediate labels
- Prototyping and iteration speed matters

**Use Modular When:**
- Debugging a specific failing step
- You have high-quality intermediate labels
- Steps are independent/reusable
- Quality control is critical (medical, legal, financial)
- You need to understand each component's behavior

**Use Hybrid When:**
- Building production systems
- You need both speed and quality
- Some steps are weak, others are strong
- You want the best of both worlds

### Implementation Tips

```python
# Profile which predictor is failing
def profile_predictors(program, testset):
    results = {}
    for name, predictor in program.named_predictors():
        # Temporarily isolate this predictor
        correct = 0
        for example in testset:
            try:
                # Test just this predictor
                pred = predictor(**example.inputs())
                if validate(example, pred):
                    correct += 1
            except:
                pass
        results[name] = correct / len(testset)
    return results

# Use results to decide which predictors need separate optimization
scores = profile_predictors(compiled, testset)
print(scores)  # {'predict1': 0.95, 'predict2': 0.60, 'predict3': 0.90}
# → predict2 needs targeted optimization!
```

### Key Insight

The magic of DSPy's bootstrap is **automatic decomposition**:
- You provide: `question → answer`
- Bootstrap traces: `question → thought → analysis → answer`
- Each predictor learns: Its specific input→output mapping

This means you get modular demos from end-to-end examples, but you can still optimize separately when needed!

---

## Summary

Building a custom optimizer requires:
1. Understanding the `.compile()` interface
2. Using `deepcopy()` and `named_predictors()`
3. Implementing your selection/generation strategy
4. Setting `predictor.demos` (and optionally instructions)
5. Marking `_compiled = True`

For multi-predictor programs:
- Start with end-to-end optimization for speed
- Profile to find weak predictors
- Use modular optimization for targeted fixes
- Re-optimize end-to-end for coherence

The beauty of DSPy's design is you can experiment with any optimization strategy while maintaining the same interface!
