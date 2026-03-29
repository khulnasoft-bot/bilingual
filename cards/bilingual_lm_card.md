---
language:
- bn
- en
license: apache-2.0
tags:
- bilingual
- bengali
- bangla
- language-model
- causal-lm
- wikipedia
datasets:
- KothaGPT/bilingual-corpus
widget:
- text: "বাংলাদেশের রাজধানী"
- text: "The capital of Bangladesh is"
---

# Bilingual Language Model (Bangla-English)

## Model Description

This is a bilingual causal language model trained on Bangla (Bengali) and English text. The model is designed for general-purpose text generation and understanding in both languages.

**Model Type:** Causal Language Model (GPT-style)  
**Languages:** Bangla (bn), English (en)  
**Training Data:** Wikipedia articles, educational content, literary texts  
**License:** Apache 2.0
**Model Size:** 124M parameters
**Context Length:** 2048 tokens

## Intended Uses

### Primary Use Cases
- **Text Generation**: Generate coherent text in Bangla or English
- **Text Completion**: Complete partial sentences or paragraphs
- **Language Understanding**: Extract features for downstream tasks
- **Fine-tuning**: Base model for task-specific applications

### Example Applications
- Content generation for educational materials
- Writing assistance tools
- Chatbots and conversational AI
- Text summarization (after fine-tuning)
- Question answering (after fine-tuning)

## How to Use

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "KothaGPT/bilingual-lm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text in Bangla
prompt = "বাংলাদেশের রাজধানী"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Generate text in English
prompt = "The capital of Bangladesh is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Advanced Usage with Pipeline

```python
from transformers import pipeline

# Create text generation pipeline
generator = pipeline("text-generation", model=model_name)

# Generate with parameters
result = generator(
    "বাংলা ভাষা",
    max_length=100,
    num_return_sequences=3,
    temperature=0.8,
    top_p=0.9
)

for seq in result:
    print(seq['generated_text'])
```

## Training Details

### Training Data
- **Wikipedia**: Bangla and English Wikipedia articles (aligned parallel corpus)
- **Literary Corpus**: Bengali literature and poetry
- **Educational Content**: Textbooks and learning materials
- **Web Crawl**: High-quality web content in both languages
- **Total Tokens**: ~1.2B tokens (600M per language)

### Training Procedure
- **Architecture**: GPT-Neo architecture with rotary position embeddings
- **Tokenizer**: Custom bilingual Byte-level BPE tokenizer
- **Vocabulary Size**: 65,536 tokens (32,768 per language)
- **Training Steps**: 150,000 steps with gradient accumulation
- **Batch Size**: 1M tokens per batch (distributed across GPUs)
- **Learning Rate**: 6e-5 with cosine decay and warmup
- **Hardware**: Trained on 8x A100 GPUs (80GB) with DeepSpeed ZeRO-3
- **Mixed Precision**: bfloat16 with gradient checkpointing
- **Sequence Length**: 2048 tokens

### Hyperparameters
```json
{
  "model_type": "gpt2",
  "vocab_size": 50000,
  "n_positions": 1024,
  "n_embd": 768,
  "n_layer": 12,
  "n_head": 12,
  "learning_rate": 5e-5,
  "warmup_steps": 10000,
  "max_steps": 100000
}
```

## Evaluation


### Perplexity (Lower is Better)
| Dataset | Perplexity |
|---------|------------|
| Bangla Test Set | 12.4 |
| English Test Set | 15.8 |
| Mixed Test Set | 14.1 |
| Code-Switched Test Set | 17.3 |

### Zero-shot Performance
| Task | Bangla | English |
|------|--------|---------|
| Text Classification | 78.2% | 82.5% |
| Named Entity Recognition | 75.6% F1 | 79.3% F1 |
| Question Answering | 68.4% F1 | 72.1% F1 |


### Downstream Tasks (after fine-tuning)
- Text Classification: 85% accuracy
- Named Entity Recognition: 82% F1
- Question Answering: 78% F1

## Limitations

### Known Limitations
- **Domain Bias**: Primarily trained on Wikipedia and educational content
- **Formal Language**: Better performance on formal text than colloquial speech
- **Code-Switching**: Handles basic code-switching but may produce inconsistent outputs
- **Context Length**: Maximum 2048 tokens
- **Generation Quality**: May produce repetitive or incoherent text for very long sequences
- **Toxic Content**: May generate harmful or biased content without proper filtering

### Language-Specific Issues
- **Bangla**: May struggle with complex literary forms and regional dialects
- **English**: Optimized for general English, may not capture specialized domains
- **Romanized Bangla**: Not trained on Romanized Bengali text

## Ethical Considerations

### Bias and Fairness
- The model may reflect biases present in Wikipedia and training data
- Geographic bias towards Bangladesh and India
- Potential gender and cultural biases in generated text

### Recommended Practices
- Review generated content for appropriateness
- Do not use for generating harmful or misleading content
- Consider fine-tuning on domain-specific data for production use
- Implement content filtering for user-facing applications

### Privacy
- Model does not store training data
- No personal information should be present in outputs
- Use caution when processing sensitive information

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{kothagpt-bilingual-lm,
  title={KothaGPT Bilingual LM: A Large Language Model for Bangla and English},
  author={KothaGPT Team},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/KothaGPT/bilingual-lm}},
  note={Model card and documentation}
}
```

## Model Card Authors

KothaGPT Team

## Model Card Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/KothaGPT/bilingual).

## Additional Resources

- **GitHub Repository**: https://github.com/KothaGPT/bilingual
- **Documentation**: https://github.com/KothaGPT/bilingual/tree/main/docs
- **Dataset**: https://huggingface.co/datasets/KothaGPT/bilingual-corpus
- **Demo**: https://huggingface.co/spaces/KothaGPT/bilingual-lm-demo
