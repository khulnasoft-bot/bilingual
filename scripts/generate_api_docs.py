"""
Generate simplified API documentation for KothaGPT.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def generate_docs():
    output_path = Path(__file__).parent.parent / "DEVELOPER_API_REFERENCE.md"
    
    docs = """# KothaGPT Developer API Reference

## 🚀 Model Management (`bilingual.models.manager`)

The `ModelManager` is a singleton responsible for lifecycle management.

### Key Methods:
- `load_model(name, version=None, model_type="causal", load_in_8bit=True)`
- `predict_batch(model_name, inputs, task=None, batch_size=None)`
- `get_pipeline(task, model_name, version=None)`
- `warmup(model_list)`

## 🧩 High-Level API (`bilingual.api`)

Simplified access to common NLP tasks.

### Methods:
- `generate(prompt, model_name="bilingual-small-lm", version=None, **kwargs)`
- `translate(text, src="bn", tgt="en", version=None, **kwargs)`
- `tokenize(text, tokenizer=None, return_ids=False)`
- `normalize(text, lang=None)`
- `batch_process(texts, operation, **kwargs)` — *Vectorized for efficiency*

## 🛡️ Security & Observability
- **Metrics**: Prometheus scraper at `:8000/metrics`
- **Validation**: Inputs capped at 5000 chars.
- **Safety**: Lexicon-based fallbacks + ML-based safety checks.

## 📊 Deployment
- **Device Support**: CUDA, MPS (Apple Silicon), XPU (Intel), CPU.
- **Quantization**: 8-bit support enabled by default on CUDA.
"""
    
    with open(output_path, "w") as f:
        f.write(docs)
    
    print(f"✅ API Documentation generated at {output_path}")

if __name__ == "__main__":
    generate_docs()
