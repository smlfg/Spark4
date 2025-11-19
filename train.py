from unsloth import FastLanguageModel  # IMPORT ORDER IS KING - must be first!
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os
import sys

"""
Production Training Script with Unsloth
- Uses Unsloth for fast, memory-efficient training
- Loads data from local JSONL file
- Implements proper error handling
- Saves adapter model to outputs/
"""

def format_prompt(sample):
    """Format training samples into prompt template"""
    instruction = sample["instruction"]
    input_text = sample["input"]
    output = sample["output"]

    if input_text.strip():
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return {"text": prompt}


def main():
    print("=" * 60)
    print("🚀 Unsloth Training Pipeline")
    print("=" * 60)

    # Configuration
    MODEL_NAME = "unsloth/tinyllama-bnb-2b"  # Fast-loading dummy model
    DATA_FILE = "dummy.jsonl"
    OUTPUT_DIR = "outputs"
    MAX_SEQ_LENGTH = 512
    MAX_STEPS = 10
    BATCH_SIZE = 1

    try:
        # Step 1: Load Model with Unsloth
        print(f"\n[1/6] Loading model: {MODEL_NAME}")
        print("      Using Unsloth for optimized training...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
        )

        print("      ✅ Model loaded successfully")

        # Step 2: Setup LoRA with Unsloth
        print("\n[2/6] Configuring LoRA adapter")

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.0,  # Optimized for speed
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth-specific optimization
            random_state=42,
        )

        print("      ✅ LoRA configured")

        # Step 3: Load Data from Local JSONL
        print(f"\n[3/6] Loading data from: {DATA_FILE}")

        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(
                f"Data file '{DATA_FILE}' not found!\n"
                f"Please run: python generate_dummy_data.py"
            )

        # Load dataset from local JSONL file
        dataset = load_dataset("json", data_files=DATA_FILE, split="train")
        print(f"      📊 Loaded {len(dataset)} samples")

        # Format dataset
        dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
        print("      ✅ Data formatted")

        # Step 4: Configure Training
        print(f"\n[4/6] Setting up trainer (max_steps={MAX_STEPS})")

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            warmup_steps=2,
            max_steps=MAX_STEPS,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            report_to="none",  # Disable wandb/tensorboard
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            args=training_args,
        )

        print("      ✅ Trainer configured")

        # Step 5: Train with Error Handling
        print("\n[5/6] Starting training...")
        print("      " + "-" * 50)

        try:
            trainer.train()
            print("      " + "-" * 50)
            print("      ✅ Training completed successfully!")

        except Exception as train_error:
            print(f"\n❌ Training failed with error:")
            print(f"   {type(train_error).__name__}: {train_error}")
            raise

        # Step 6: Save Model
        print(f"\n[6/6] Saving adapter model to: {OUTPUT_DIR}/")

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print("      ✅ Model saved successfully")

        print("\n" + "=" * 60)
        print("🎉 Training Pipeline Complete!")
        print("=" * 60)
        print(f"📁 Output location: {OUTPUT_DIR}/")
        print(f"📊 Training steps: {MAX_STEPS}")
        print(f"💾 Model type: LoRA Adapter")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Fatal Error:")
        print(f"   {type(e).__name__}: {e}")
        print("\n🔍 Troubleshooting:")
        print("   1. Check if dummy.jsonl exists (run generate_dummy_data.py)")
        print("   2. Verify CUDA/GPU availability")
        print("   3. Check dependencies: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
