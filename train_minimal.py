#!/usr/bin/env python3
"""
Minimal LLaMA Training Script
- Loads smallest LLaMA model (TinyLlama-1.1B)
- Creates 10-line dummy dataset
- Trains for 1 epoch
- Saves LoRA adapter
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def main():
    print("=" * 50)
    print("Minimal LLaMA Training Script")
    print("=" * 50)

    # 1. Load Model and Tokenizer (TinyLlama is smallest available)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\n[1/5] Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # 2. Setup LoRA
    print("\n[2/5] Configuring LoRA adapter")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Create Dummy Dataset (10 lines)
    print("\n[3/5] Creating dummy dataset (10 samples)")
    dummy_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Deep learning models require large amounts of data.",
        "Natural language processing enables computers to understand text.",
        "LLaMA is a large language model developed by Meta.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
        "LoRA is an efficient method for model adaptation.",
        "Transformers revolutionized natural language processing.",
        "Training neural networks requires careful hyperparameter tuning."
    ]

    # Tokenize the dataset
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    dataset = Dataset.from_dict({"text": dummy_texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    print(f"Dataset size: {len(tokenized_dataset)} samples")

    # 4. Training Configuration
    print("\n[4/5] Starting training (1 epoch)")
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 5. Train and Save
    trainer.train()

    print("\n[5/5] Saving adapter")
    model.save_pretrained("./lora_adapter")
    tokenizer.save_pretrained("./lora_adapter")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("Adapter saved to: ./lora_adapter")
    print("=" * 50)

if __name__ == "__main__":
    main()
