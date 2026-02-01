# Caching Mechanism in Python

## Table of Contents
1. [What is Caching?](#what-is-caching)
2. [Built-in Python Caching](#built-in-python-caching)
3. [7 Critical Concepts](#7-critical-concepts)
4. [Custom Cache Implementation](#custom-cache-implementation)
5. [DSPy Use Cases](#dspy-use-cases)
6. [Best Practices](#best-practices)

---

## What is Caching?

**Caching** = Store expensive computation results to avoid recomputing them.

```python
# Without cache - slow
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

fibonacci(35)  # Takes ~5 seconds, recalculates same values repeatedly

# With cache - fast
from functools import cache

@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

fibonacci(35)  # Takes <0.01 seconds, each value calculated once
```

**When to use:**
- Expensive computations (API calls, DB queries, complex calculations)
- Pure functions (same input → same output)
- Frequently accessed data

**When NOT to use:**
- Random/time-dependent results
- Frequently changing data
- Large memory footprint

---

## Built-in Python Caching

### 1. @lru_cache - Most Common

```python
from functools import lru_cache

@lru_cache(maxsize=128)  # Keep 128 most recent results
def get_user(user_id: int):
    print(f"Fetching user {user_id}...")
    return fetch_from_database(user_id)

# First call - hits database
get_user(1)  # Fetching user 1... → {"name": "Alice"}

# Second call - uses cache
get_user(1)  # (no print) → {"name": "Alice"}

# Check cache statistics
get_user.cache_info()
# CacheInfo(hits=1, misses=1, maxsize=128, currsize=1)

# Clear cache
get_user.cache_clear()
```

**LRU = Least Recently Used** - evicts oldest unused items when cache is full.

### 2. @cache - Unlimited Cache (Python 3.9+)

```python
from functools import cache

@cache  # No size limit - use carefully!
def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n-1)

factorial(100)  # All intermediate results cached
```

### 3. Comparison

| Decorator | Size Limit | Eviction | Use Case |
|-----------|------------|----------|----------|
| `@lru_cache(maxsize=N)` | N items | LRU | Most cases |
| `@lru_cache(maxsize=None)` | Unlimited | None | Small result set |
| `@cache` | Unlimited | None | Cleaner syntax (3.9+) |

---

## 7 Critical Concepts

### 1. Cache Key Design

**Problem:** Only hashable types can be cache keys.

```python
# ❌ BAD - unhashable arguments
@lru_cache
def process(items: list):  # TypeError: unhashable type: 'list'
    return sum(items)

@lru_cache
def get_data(filters: dict):  # TypeError: unhashable type: 'dict'
    return query_db(filters)

# ✅ GOOD - convert to hashable
@lru_cache
def process(items: tuple):  # tuple is hashable
    return sum(items)

process((1, 2, 3))  # Works!

# ✅ GOOD - manual conversion
def get_data(filters: dict):
    cache_key = tuple(sorted(filters.items()))
    return _cached_get_data(cache_key)

@lru_cache
def _cached_get_data(cache_key: tuple):
    filters = dict(cache_key)
    return query_db(filters)
```

**Hashable types:** int, str, tuple, frozenset  
**Unhashable types:** list, dict, set

### 2. Memory Management

```python
# ❌ BAD - memory leak
cache = {}  # Grows forever!

def get_data(key):
    if key not in cache:
        cache[key] = expensive_operation(key)
    return cache[key]

# ✅ GOOD - size limit with LRU eviction
from functools import lru_cache

@lru_cache(maxsize=128)  # Auto-evicts when full
def get_data(key):
    return expensive_operation(key)

# ✅ GOOD - manual size control
from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size=100):
        self._cache = OrderedDict()
        self._max_size = max_size
    
    def get(self, key):
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def set(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        
        # Evict oldest if over limit
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
```

### 3. Cache Invalidation

**Invalidation** = Remove cached data when it becomes stale.

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_user(user_id: int):
    return fetch_from_db(user_id)

# User data cached
get_user(1)  # → {"name": "Alice"}

# Update user in database
update_user_in_db(1, name="Bob")

# Cache still returns old data!
get_user(1)  # → {"name": "Alice"} ❌ Stale!

# Invalidate cache
get_user.cache_clear()

# Now fetches fresh data
get_user(1)  # → {"name": "Bob"} ✅
```

**Invalidation Strategies:**

#### Time-Based (TTL - Time To Live)
```python
import time

class TTLCache:
    def __init__(self, ttl_seconds=60):
        self._cache = {}
        self._timestamps = {}
        self._ttl = ttl_seconds
    
    def get(self, key):
        if key in self._cache:
            # Check if expired
            if time.time() - self._timestamps[key] < self._ttl:
                return self._cache[key]
            else:
                # Expired - remove
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key, value):
        self._cache[key] = value
        self._timestamps[key] = time.time()

# Usage
cache = TTLCache(ttl_seconds=300)  # 5 minutes
cache.set("user:1", {"name": "Alice"})
cache.get("user:1")  # Valid for 5 minutes
```

#### Event-Based (Write-Through)
```python
class CacheWithInvalidation:
    def __init__(self):
        self._cache = {}
    
    def get_user(self, user_id):
        if user_id not in self._cache:
            self._cache[user_id] = fetch_from_db(user_id)
        return self._cache[user_id]
    
    def update_user(self, user_id, data):
        # Update database
        save_to_db(user_id, data)
        # Invalidate cache
        self._cache.pop(user_id, None)
```

#### Selective Invalidation
```python
class SelectiveCache:
    def __init__(self):
        self._cache = {}
    
    @lru_cache(maxsize=100)
    def get_user(self, user_id):
        return fetch_from_db(user_id)
    
    def invalidate_user(self, user_id):
        # Problem: @lru_cache only has cache_clear() (clears ALL)
        # Solution: Use manual cache with selective deletion
        pass

# Better approach - manual cache
class SelectiveCache:
    def __init__(self):
        self._cache = {}
    
    def get_user(self, user_id):
        if user_id not in self._cache:
            self._cache[user_id] = fetch_from_db(user_id)
        return self._cache[user_id]
    
    def invalidate_user(self, user_id):
        self._cache.pop(user_id, None)  # Remove only this user
```

### 4. Thread Safety

```python
# ❌ NOT thread-safe
cache = {}

def get_data(key):
    if key not in cache:  # Race condition!
        cache[key] = expensive_operation(key)
    return cache[key]

# Two threads can both see key not in cache and compute twice!

# ✅ Thread-safe with lock
import threading

cache = {}
lock = threading.Lock()

def get_data(key):
    with lock:
        if key not in cache:
            cache[key] = expensive_operation(key)
        return cache[key]

# ✅ @lru_cache is thread-safe by default
from functools import lru_cache

@lru_cache(maxsize=128)  # Thread-safe!
def get_data(key):
    return expensive_operation(key)
```

### 5. Cache Monitoring

```python
class MonitoredCache:
    def __init__(self):
        self._cache = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        
        self.misses += 1
        value = expensive_operation(key)
        self._cache[key] = value
        return value
    
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate(),
            "size": len(self._cache)
        }

# Usage
cache = MonitoredCache()
cache.get("key1")
cache.get("key1")  # Hit
cache.get("key2")
print(cache.stats())
# {"hits": 1, "misses": 2, "hit_rate": 0.33, "size": 2}

# With @lru_cache
@lru_cache(maxsize=128)
def get_data(key):
    return expensive_operation(key)

get_data("key1")
get_data("key1")
print(get_data.cache_info())
# CacheInfo(hits=1, misses=1, maxsize=128, currsize=1)
```

### 6. Persistent Cache

```python
import pickle
from pathlib import Path

class PersistentCache:
    def __init__(self, cache_file="cache.pkl"):
        self.cache_file = Path(cache_file)
        self._cache = self._load()
    
    def _load(self):
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        return {}
    
    def _save(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self._cache, f)
    
    def get(self, key):
        return self._cache.get(key)
    
    def set(self, key, value):
        self._cache[key] = value
        self._save()  # Persist immediately
    
    def clear(self):
        self._cache.clear()
        self._save()

# Usage
cache = PersistentCache("my_cache.pkl")
cache.set("key1", "value1")
# Survives program restart!
```

### 7. Cache Warming

**Cache warming** = Pre-populate cache with frequently accessed data.

```python
class WarmedCache:
    def __init__(self):
        self._cache = {}
        self._warm_cache()
    
    def _warm_cache(self):
        # Pre-load frequently accessed data
        popular_ids = [1, 2, 3, 5, 10]
        for user_id in popular_ids:
            self._cache[user_id] = fetch_from_db(user_id)
        print(f"Cache warmed with {len(popular_ids)} items")
    
    def get_user(self, user_id):
        if user_id not in self._cache:
            self._cache[user_id] = fetch_from_db(user_id)
        return self._cache[user_id]
```

---

## Custom Cache Implementation

### Simple Dictionary Cache

```python
class SimpleCache:
    def __init__(self):
        self._cache = {}
    
    def get(self, key, default=None):
        return self._cache.get(key, default)
    
    def set(self, key, value):
        self._cache[key] = value
    
    def clear(self):
        self._cache.clear()
    
    def size(self):
        return len(self._cache)
```

### LRU Cache from Scratch

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size=100):
        self._cache = OrderedDict()
        self._max_size = max_size
    
    def get(self, key):
        if key not in self._cache:
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]
    
    def set(self, key, value):
        if key in self._cache:
            # Update and move to end
            self._cache.move_to_end(key)
        self._cache[key] = value
        
        # Evict oldest if over limit
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)  # Remove first (oldest)
    
    def clear(self):
        self._cache.clear()

# Usage
cache = LRUCache(max_size=3)
cache.set("a", 1)
cache.set("b", 2)
cache.set("c", 3)
cache.set("d", 4)  # Evicts "a"
print(cache.get("a"))  # None
print(cache.get("b"))  # 2
```

### Production-Ready Cache

```python
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

class ProductionCache:
    def __init__(self, max_size=100, ttl_seconds=None):
        self._cache = OrderedDict()
        self._timestamps = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            # Check TTL
            if self._ttl and time.time() - self._timestamps[key] > self._ttl:
                del self._cache[key]
                del self._timestamps[key]
                self.misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
    
    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
            
            # Evict oldest if over limit
            if len(self._cache) > self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
    
    def invalidate(self, key: str):
        with self._lock:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "size": len(self._cache),
            "max_size": self._max_size
        }
```

---

## DSPy Use Cases

### 1. Caching LLM Responses

```python
from functools import lru_cache
import hashlib

class CachedLLM:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
    
    @lru_cache(maxsize=1000)
    def _call_cached(self, prompt_hash: str, temperature: float):
        # Actual LLM API call
        return call_openai_api(prompt_hash, temperature, self.model)
    
    def __call__(self, prompt: str, temperature: float = 0.7):
        # Hash prompt for cache key (prompts can be long)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return self._call_cached(prompt_hash, temperature)
    
    def clear_cache(self):
        self._call_cached.cache_clear()
    
    def cache_stats(self):
        info = self._call_cached.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "hit_rate": info.hits / (info.hits + info.misses) if info.hits + info.misses > 0 else 0
        }

# Usage
llm = CachedLLM()
response1 = llm("What is Python?")  # API call
response2 = llm("What is Python?")  # Cached!
print(llm.cache_stats())
# {"hits": 1, "misses": 1, "size": 1, "hit_rate": 0.5}
```

### 2. Caching DSPy Predictor Results

```python
import dspy
from functools import lru_cache

class CachedPredictor:
    def __init__(self, signature):
        self.predictor = dspy.Predict(signature)
    
    @lru_cache(maxsize=500)
    def _predict_cached(self, **kwargs):
        # Convert kwargs to hashable tuple
        cache_key = tuple(sorted(kwargs.items()))
        return self.predictor(**dict(cache_key))
    
    def __call__(self, **kwargs):
        return self._predict_cached(**kwargs)
    
    def clear_cache(self):
        self._predict_cached.cache_clear()

# Usage
predictor = CachedPredictor("question -> answer")
result1 = predictor(question="What is 2+2?")  # LLM call
result2 = predictor(question="What is 2+2?")  # Cached!
```

### 3. Caching Optimization Results

```python
import dspy
from functools import lru_cache
import pickle

class CachedOptimizer:
    def __init__(self, cache_file="optimizer_cache.pkl"):
        self.cache_file = cache_file
        self._cache = self._load_cache()
    
    def _load_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_cache(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self._cache, f)
    
    def optimize(self, program, trainset, metric):
        # Create cache key from program signature and trainset hash
        cache_key = (
            program.__class__.__name__,
            hash(tuple(str(x) for x in trainset[:5]))  # Sample hash
        )
        
        if cache_key in self._cache:
            print("Loading optimized program from cache...")
            return self._cache[cache_key]
        
        # Run optimization
        optimizer = dspy.BootstrapFewShot(metric=metric)
        optimized = optimizer.compile(program, trainset=trainset)
        
        # Cache result
        self._cache[cache_key] = optimized
        self._save_cache()
        
        return optimized

# Usage
optimizer = CachedOptimizer()
optimized_program = optimizer.optimize(my_program, trainset, metric)
# Second run loads from cache instantly!
```

### 4. Caching Retrieval Results

```python
from functools import lru_cache
import dspy

class CachedRetriever:
    def __init__(self, retriever):
        self.retriever = retriever
    
    @lru_cache(maxsize=1000)
    def _retrieve_cached(self, query: str, k: int):
        return tuple(self.retriever(query, k=k))  # Convert to tuple (hashable)
    
    def __call__(self, query: str, k: int = 5):
        return list(self._retrieve_cached(query, k))  # Convert back to list
    
    def clear_cache(self):
        self._retrieve_cached.cache_clear()

# Usage with DSPy
rm = dspy.ColBERTv2(url="http://localhost:8893/api/search")
cached_rm = CachedRetriever(rm)

# First call - hits retrieval system
results1 = cached_rm("What is Python?", k=3)

# Second call - cached
results2 = cached_rm("What is Python?", k=3)
```

---

## Best Practices

### ✅ DO

1. **Use @lru_cache for simple cases**
   ```python
   @lru_cache(maxsize=128)
   def expensive_function(x):
       return x ** 2
   ```

2. **Set appropriate maxsize**
   ```python
   # Small dataset
   @lru_cache(maxsize=32)
   
   # Medium dataset
   @lru_cache(maxsize=128)
   
   # Large dataset
   @lru_cache(maxsize=1024)
   ```

3. **Monitor cache performance**
   ```python
   func.cache_info()  # Check hit rate
   ```

4. **Use hashable arguments**
   ```python
   @lru_cache
   def process(items: tuple):  # ✅ tuple
       return sum(items)
   ```

5. **Invalidate when data changes**
   ```python
   def update_data(key, value):
       save_to_db(key, value)
       get_data.cache_clear()  # Invalidate
   ```

6. **Add TTL for time-sensitive data**
   ```python
   cache = TTLCache(ttl_seconds=300)  # 5 minutes
   ```

### ❌ DON'T

1. **Don't cache random/time-dependent results**
   ```python
   @lru_cache  # ❌ BAD
   def get_random():
       return random.random()
   
   @lru_cache  # ❌ BAD
   def get_current_time():
       return datetime.now()
   ```

2. **Don't use unlimited cache carelessly**
   ```python
   @lru_cache(maxsize=None)  # ❌ Memory leak risk
   def load_large_file(path):
       return open(path).read()
   ```

3. **Don't cache with mutable arguments**
   ```python
   @lru_cache  # ❌ TypeError
   def process(items: list):
       return sum(items)
   ```

4. **Don't forget thread safety**
   ```python
   # ❌ Race condition
   if key not in cache:
       cache[key] = expensive()
   
   # ✅ Use lock or @lru_cache
   ```

5. **Don't cache sensitive data**
   ```python
   @lru_cache  # ❌ Security risk
   def get_password(user_id):
       return fetch_password(user_id)
   ```

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **Cache Key** | Only hashable types (int, str, tuple) |
| **Memory** | Set maxsize to prevent memory leaks |
| **Invalidation** | Clear cache when data changes |
| **Thread Safety** | Use locks or @lru_cache |
| **Monitoring** | Track hit rate with cache_info() |
| **Persistence** | Use pickle for cross-session caching |
| **TTL** | Add expiration for time-sensitive data |

**Golden Rule:** Cache expensive, pure functions with immutable arguments and appropriate size limits.


---

## DSPy's Production Caching System

DSPy implements a sophisticated **2-tier caching system** for LLM calls. This is a real-world example of production-grade caching.

### Architecture

```
LLM Request → Memory Cache (LRU) → Disk Cache (FanoutCache) → API Call
              ↑ Fast (RAM)          ↑ Persistent (Disk)        ↑ Slow/Costly
              ↑ ~1ms                ↑ ~10ms                    ↑ ~1000ms
```

### The Cache Class

```python
from cachetools import LRUCache
from diskcache import FanoutCache
import threading

class Cache:
    def __init__(
        self,
        enable_disk_cache: bool,
        enable_memory_cache: bool,
        disk_cache_dir: str,
        disk_size_limit_bytes: int = 1024 * 1024 * 10,  # 10MB
        memory_max_entries: int = 1000000,
    ):
        # Tier 1: In-memory LRU cache (fast, volatile)
        if enable_memory_cache:
            self.memory_cache = LRUCache(maxsize=memory_max_entries)
        else:
            self.memory_cache = {}
        
        # Tier 2: Disk cache (slower, persistent)
        if enable_disk_cache:
            self.disk_cache = FanoutCache(
                shards=16,           # Split into 16 files for concurrency
                timeout=10,
                directory=disk_cache_dir,
                size_limit=disk_size_limit_bytes,
            )
        else:
            self.disk_cache = {}
        
        self._lock = threading.RLock()  # Thread-safe
```

**Why 2 tiers?**
- **Memory**: Lightning fast (~1ms), but lost on restart
- **Disk**: Survives restarts (~10ms), shareable across runs
- **Best of both worlds**: Speed + persistence

**Why FanoutCache with 16 shards?**
- Splits cache into 16 separate files
- Reduces lock contention in multi-threaded scenarios
- Better concurrent read/write performance

### Cache Key Generation

The most critical part - how to uniquely identify a request:

```python
from hashlib import sha256
import orjson
import pydantic

def cache_key(self, request: dict, ignored_args_for_cache_key: list[str]) -> str:
    """Generate unique cache key by hashing request JSON."""
    
    def transform_value(value):
        # Handle Pydantic models
        if isinstance(value, pydantic.BaseModel):
            return value.model_dump(mode="json")
        
        # Handle callables (functions)
        elif callable(value):
            try:
                return f"<callable_source:{inspect.getsource(value)}>"
            except (TypeError, OSError):
                return f"<callable:{value.__name__}>"
        
        # Recursively handle dicts
        elif isinstance(value, dict):
            return {k: transform_value(v) for k, v in value.items()}
        
        else:
            return value
    
    # Filter out ignored args (like api_key)
    params = {
        k: transform_value(v) 
        for k, v in request.items() 
        if k not in ignored_args_for_cache_key
    }
    
    # Hash the sorted JSON representation
    return sha256(
        orjson.dumps(params, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()
```

**Step-by-step example:**

```python
request = {
    "model": "openai/gpt-4",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "temperature": 0.0,
    "max_tokens": 100,
    "api_key": "sk-xxx",  # Will be ignored
}

ignored = ["api_key", "api_base", "base_url"]

# Step 1: Filter ignored args
params = {
    "model": "openai/gpt-4",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "temperature": 0.0,
    "max_tokens": 100,
}

# Step 2: Sort keys and serialize to JSON (orjson is faster than json)
json_bytes = orjson.dumps(params, option=orjson.OPT_SORT_KEYS)
# b'{"max_tokens":100,"messages":[{"content":"What is AI?","role":"user"}],"model":"openai/gpt-4","temperature":0.0}'

# Step 3: SHA256 hash
cache_key = "a3f5b8c9d2e1f4a7b6c5d8e9f0a1b2c3..."  # 64 character hex string
```

**Why ignore certain args?**
- `api_key`: Same request with different API keys should hit same cache
- `api_base`: Different endpoints but same model/params should cache together
- Security: Don't include secrets in cache keys

**Why sort keys?**
```python
# Without sorting - different hashes!
{"a": 1, "b": 2}  # → hash_1
{"b": 2, "a": 1}  # → hash_2 (different!)

# With sorting - same hash
orjson.dumps({"a": 1, "b": 2}, option=orjson.OPT_SORT_KEYS)  # → hash_x
orjson.dumps({"b": 2, "a": 1}, option=orjson.OPT_SORT_KEYS)  # → hash_x (same!)
```

### Cache Lookup (get)

```python
def get(self, request: dict, ignored_args_for_cache_key: list[str]) -> Any:
    """Retrieve from cache with 2-tier lookup."""
    
    # Generate cache key
    try:
        key = self.cache_key(request, ignored_args_for_cache_key)
    except Exception:
        logger.debug(f"Failed to generate cache key for request: {request}")
        return None
    
    # Tier 1: Check memory cache (fast)
    if self.enable_memory_cache and key in self.memory_cache:
        with self._lock:
            response = self.memory_cache[key]
    
    # Tier 2: Check disk cache (slower)
    elif self.enable_disk_cache and key in self.disk_cache:
        response = self.disk_cache[key]
        
        # Promote to memory cache for next time
        if self.enable_memory_cache:
            with self._lock:
                self.memory_cache[key] = response
    
    else:
        return None  # Cache miss
    
    # Deep copy to prevent mutations
    response = copy.deepcopy(response)
    
    # Mark as cache hit and clear usage
    if hasattr(response, "usage"):
        response.usage = {}  # No tokens used on cache hit
        response.cache_hit = True
    
    return response
```

**The lookup flow:**

```
Request: "What is AI?"
    ↓
Generate cache key: "a3f5b8c9..."
    ↓
Check memory_cache[key]
    ↓ Hit? Return immediately (1ms) ✅
    ↓ Miss? Continue...
    ↓
Check disk_cache[key]
    ↓ Hit? Promote to memory + return (10ms) ✅
    ↓ Miss? Return None (cache miss) ❌
```

**Why deep copy?**
```python
# Without deep copy - mutation bug!
cached = cache.get(request)
cached["messages"].append({"role": "user", "content": "more"})
# Next cache hit returns mutated data! 💥

# With deep copy - safe
cached = copy.deepcopy(cache.get(request))
cached["messages"].append(...)  # Original cache unchanged ✅
```

**Why clear usage on cache hit?**
```python
# API response has usage data
response.usage = {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
}

# Cache hit - no API call made
cached_response.usage = {}  # Clear to avoid double-counting costs
cached_response.cache_hit = True  # Mark for tracking
```

### Cache Storage (put)

```python
def put(
    self,
    request: dict,
    value: Any,
    ignored_args_for_cache_key: list[str],
    enable_memory_cache: bool = True,
) -> None:
    """Store value in cache (both tiers)."""
    
    enable_memory_cache = self.enable_memory_cache and enable_memory_cache
    
    # Early return if both disabled
    if not enable_memory_cache and not self.enable_disk_cache:
        return
    
    # Generate cache key
    try:
        key = self.cache_key(request, ignored_args_for_cache_key)
    except Exception:
        logger.debug(f"Failed to generate cache key")
        return
    
    # Store in memory cache
    if enable_memory_cache:
        with self._lock:
            self.memory_cache[key] = value
    
    # Store in disk cache
    if self.enable_disk_cache:
        try:
            self.disk_cache[key] = value
        except Exception as e:
            # Disk full or value not picklable
            logger.debug(f"Failed to put value in disk cache: {e}")
```

**Why try/except for disk?**
- Disk might be full
- Value might not be picklable (some objects can't be serialized)
- Don't crash the program if disk cache fails

**Why `enable_memory_cache` parameter?**
```python
# Normal case - cache in both tiers
cache.put(request, result, ignored_args)

# Special case - only disk cache (avoid memory growth)
cache.put(request, result, ignored_args, enable_memory_cache=False)
```

### The Decorator Pattern

This is where DSPy applies caching transparently:

```python
def request_cache(
    cache_arg_name: str = None,
    ignored_args_for_cache_key: list[str] = None,
    enable_memory_cache: bool = True,
):
    """Decorator for applying caching to a function."""
    
    ignored_args_for_cache_key = ignored_args_for_cache_key or ["api_key", "api_base", "base_url"]
    
    def decorator(fn):
        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            import dspy
            
            # Build cache request
            if cache_arg_name:
                modified_request = copy.deepcopy(kwargs[cache_arg_name])
            else:
                modified_request = copy.deepcopy(kwargs)
                for i, arg in enumerate(args):
                    modified_request[f"positional_arg_{i}"] = arg
            
            # Add function identifier to cache key
            fn_identifier = f"{fn.__module__}.{fn.__qualname__}"
            modified_request["_fn_identifier"] = fn_identifier
            
            # Try to get from cache
            cached_result = dspy.cache.get(modified_request, ignored_args_for_cache_key)
            if cached_result is not None:
                return cached_result  # Cache hit! 🎉
            
            # Cache miss - call actual function
            original_request = copy.deepcopy(modified_request)
            result = fn(*args, **kwargs)
            
            # Store in cache
            dspy.cache.put(original_request, result, ignored_args_for_cache_key, enable_memory_cache)
            
            return result
        
        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            # Same logic but with await
            ...
        
        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
```

**How it's used in DSPy:**

```python
# In lm.py
@request_cache(
    cache_arg_name="request",
    ignored_args_for_cache_key=["api_key", "api_base", "base_url"]
)
def litellm_completion(request: dict, num_retries: int, cache: dict):
    return litellm.completion(**request)
```

**Why include `_fn_identifier`?**

```python
# Without function identifier - collision!
def litellm_completion(request):
    return litellm.completion(**request)

def litellm_text_completion(request):
    return litellm.text_completion(**request)

# Same request dict → same cache key → wrong result returned! 💥

# With function identifier - unique keys
modified_request["_fn_identifier"] = "dspy.clients.lm.litellm_completion"
# vs
modified_request["_fn_identifier"] = "dspy.clients.lm.litellm_text_completion"
# Different cache keys ✅
```

### Real-World Usage Example

```python
import dspy

# Initialize LM with caching enabled
lm = dspy.LM("openai/gpt-4", cache=True)

# First call - cache miss
print("Call 1:")
result1 = lm("What is 2+2?")
# Flow:
# 1. Generate cache key: "a3f5b8c9..."
# 2. Check memory cache: MISS
# 3. Check disk cache: MISS
# 4. Call OpenAI API (costs $0.01, takes 1000ms)
# 5. Store in memory_cache[key] = response
# 6. Store in disk_cache[key] = response
# 7. Return result

# Second call - memory cache hit
print("Call 2:")
result2 = lm("What is 2+2?")
# Flow:
# 1. Generate cache key: "a3f5b8c9..." (same!)
# 2. Check memory cache: HIT! ✅
# 3. Return cached result (free, 1ms)

# Restart program, memory cache cleared...

# Third call - disk cache hit
print("Call 3 (after restart):")
result3 = lm("What is 2+2?")
# Flow:
# 1. Generate cache key: "a3f5b8c9..."
# 2. Check memory cache: MISS (cleared on restart)
# 3. Check disk cache: HIT! ✅
# 4. Promote to memory_cache[key] = response
# 5. Return cached result (free, 10ms)

# Fourth call - memory cache hit again
print("Call 4:")
result4 = lm("What is 2+2?")
# Flow:
# 1. Generate cache key: "a3f5b8c9..."
# 2. Check memory cache: HIT! ✅ (promoted from disk)
# 3. Return cached result (free, 1ms)
```

### Cache Invalidation Strategies

**1. Different parameters = different cache:**

```python
lm = dspy.LM("openai/gpt-4")

# Different temperature → different cache key
result1 = lm("Tell me a joke", temperature=0.0)
result2 = lm("Tell me a joke", temperature=1.0)  # Different result, not cached

# Different max_tokens → different cache key
result3 = lm("Explain AI", max_tokens=50)
result4 = lm("Explain AI", max_tokens=100)  # Different cache entry
```

**2. Using rollout_id to bypass cache:**

```python
lm = dspy.LM("openai/gpt-4", temperature=1.0)

# Same prompt, different rollout IDs → different cache entries
result1 = lm("Tell me a joke", rollout_id=1)
result2 = lm("Tell me a joke", rollout_id=2)  # New API call, different result
result3 = lm("Tell me a joke", rollout_id=1)  # Cached from first call
```

**3. Manual cache clearing:**

```python
import dspy

# Clear memory cache only
dspy.cache.reset_memory_cache()

# Save memory cache to disk
dspy.cache.save_memory_cache("my_cache.pkl")

# Load memory cache from disk
dspy.cache.load_memory_cache("my_cache.pkl", allow_pickle=True)
```

### Performance Characteristics

| Operation | Memory Cache | Disk Cache | API Call |
|-----------|--------------|------------|----------|
| Latency | ~1ms | ~10ms | ~1000ms |
| Cost | Free | Free | $0.001-0.01 |
| Persistence | No (RAM) | Yes (Disk) | N/A |
| Capacity | 1M entries | 10MB-1GB | Unlimited |

**Cache hit rates in practice:**
- Development: 80-95% (lots of repeated prompts)
- Production: 30-60% (more diverse queries)
- Testing: 95%+ (same test cases)

### Thread Safety

DSPy's cache is thread-safe:

```python
import threading

lm = dspy.LM("openai/gpt-4")

def worker():
    result = lm("What is AI?")
    print(result)

# Multiple threads can safely access cache
threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Only 1 API call made, 9 cache hits!
```

**How it's thread-safe:**
```python
# RLock (Reentrant Lock) allows same thread to acquire multiple times
self._lock = threading.RLock()

def get(self, key):
    with self._lock:  # Thread-safe access
        return self.memory_cache[key]

def put(self, key, value):
    with self._lock:  # Thread-safe access
        self.memory_cache[key] = value
```

### Key Insights

1. **2-tier design**: Memory (fast) + Disk (persistent) = best of both worlds
2. **SHA256 hashing**: Unique 64-char hex key for each unique request
3. **Ignored args**: API keys don't affect caching (security + efficiency)
4. **Deep copying**: Prevents cache corruption from mutations
5. **Thread-safe**: RLock protects concurrent access
6. **Decorator pattern**: Transparent caching without changing function code
7. **Function identifier**: Different functions with same args don't collide
8. **LRU eviction**: Automatic cleanup when memory full (1M entries default)
9. **Graceful degradation**: Disk cache failures don't crash program
10. **Cache promotion**: Disk hits promoted to memory for future speed

### Benefits in DSPy

**💰 Cost Savings:**
```python
# Without cache
for i in range(100):
    lm("What is AI?")  # 100 API calls × $0.01 = $1.00

# With cache
for i in range(100):
    lm("What is AI?")  # 1 API call + 99 cache hits = $0.01
```

**⚡ Speed Improvement:**
```python
# Without cache: 100 × 1000ms = 100 seconds
# With cache: 1000ms + 99 × 1ms = ~1.1 seconds (90x faster!)
```

**🔄 Reproducibility:**
```python
# Same input always returns same output (when temperature=0)
result1 = lm("What is 2+2?")  # "4"
result2 = lm("What is 2+2?")  # "4" (from cache, guaranteed same)
```

**💾 Persistence:**
```python
# Day 1
lm("Explain quantum computing")  # API call, cached to disk

# Day 2 (restart program)
lm("Explain quantum computing")  # Instant from disk cache!
```

This is production-grade caching that handles real-world challenges: thread safety, persistence, memory management, and graceful degradation.
