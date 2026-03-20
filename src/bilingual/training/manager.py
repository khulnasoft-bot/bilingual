"""
Training and Fine-tuning Manager for KothaGPT.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

def fine_tune_model(
    model_name: str,
    train_data: List[Dict[str, str]],
    output_dir: str,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    **kwargs,
) -> str:
    """
    Fine-tune a language model on custom data.
    """
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError:
        raise ImportError(
            "PyTorch and transformers are required for fine-tuning. "
            "Install with: pip install torch transformers"
        )

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare dataset
    class CustomDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=512):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            input_text = item["input"]
            target_text = item["output"]

            # Combine input and output for language modeling
            full_text = f"{input_text} {target_text}"

            encodings = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": encodings["input_ids"].flatten(),
                "attention_mask": encodings["attention_mask"].flatten(),
                "labels": encodings["input_ids"].flatten(),
            }

    dataset = CustomDataset(train_data, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        **kwargs,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Model fine-tuned and saved to {output_dir}")
    return output_dir
