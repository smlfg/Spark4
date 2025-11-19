# Spark4 - Production Unsloth Training Environment

Production-ready Docker environment for fine-tuning LLMs with Unsloth, TRL, and transformers.

## Features

- **Base Image**: NVIDIA PyTorch 24.01 (Python 3.10, CUDA 12.3, PyTorch 2.2.0)
- **Pinned Dependencies**: All versions locked for reproducibility
- **Pre-compiled Flash Attention 2**: Ready to use, no compilation needed
- **Multi-stage Build**: Optimized image size
- **Security**: Non-root user, minimal attack surface
- **Production Ready**: Health checks, proper caching, metadata

## Pinned Versions

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Base runtime (no Python 3.13 issues) |
| PyTorch | 2.2.0 | Deep learning framework |
| CUDA | 12.3 | GPU acceleration |
| Transformers | 4.45.2 | Hugging Face models |
| TRL | 0.11.4 | Training framework |
| Unsloth | 2024.11 | Fast fine-tuning |
| Flash Attention | 2.5.9.post1 | Efficient attention |
| PEFT | 0.13.2 | Parameter-efficient tuning |
| bitsandbytes | 0.44.1 | Quantization |

## Requirements

- Docker 20.10+
- NVIDIA Docker runtime
- NVIDIA GPU with compute capability 7.0+ (e.g., V100, T4, A100, RTX 20xx+)
- NVIDIA Driver 545+

## Quick Start

### Build the Image

```bash
docker build -t spark4-unsloth:latest .
```

Build with custom cache directory:

```bash
docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t spark4-unsloth:latest \
  .
```

### Run the Container

Basic run:

```bash
docker run --gpus all -it --rm spark4-unsloth:latest
```

With mounted directories:

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/scripts:/app/scripts \
  spark4-unsloth:latest
```

With Weights & Biases:

```bash
docker run --gpus all -it --rm \
  -e WANDB_API_KEY=your_api_key \
  -v $(pwd)/data:/app/data \
  spark4-unsloth:latest
```

### Verify Installation

Inside the container:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import unsloth; print('Unsloth ready!')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
```

## Usage Examples

### Fine-tuning with Unsloth

```python
from unsloth import FastLanguageModel
import torch

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
)

# Your training code here...
```

### Using TRL's SFTTrainer

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()
```

## Build Optimization

The Dockerfile uses multi-stage builds to reduce final image size:

1. **Builder stage**: Compiles Flash Attention and installs all dependencies
2. **Production stage**: Copies only necessary files, removes build tools

Build time optimizations:
- Flash Attention is compiled early for Docker layer caching
- Dependencies are installed in order of change frequency
- Build context excludes unnecessary files via `.dockerignore`

## Security

- Runs as non-root user `appuser` (UID 1000)
- Minimal runtime dependencies
- No unnecessary packages or build tools in production image
- Health checks ensure CUDA is available

## Troubleshooting

### CUDA not available

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker runtime
docker run --gpus all nvcr.io/nvidia/pytorch:24.01-py3 nvidia-smi
```

### Out of memory

Reduce batch size or use gradient checkpointing:

```python
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing = True,
    # ... other parameters
)
```

### Flash Attention issues

Flash Attention requires:
- CUDA 11.8+
- GPU compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- PyTorch 2.0+

All requirements are met in this image.

## Development

To add more dependencies, update the Dockerfile and rebuild:

```dockerfile
# Add after existing pip install commands
RUN pip install --no-cache-dir \
    your-package==1.0.0
```

Always pin versions for reproducibility.

## License

This Dockerfile and configuration are provided as-is for the Spark4 project.

## Support

For issues related to:
- **Unsloth**: https://github.com/unslothai/unsloth
- **TRL**: https://github.com/huggingface/trl
- **Transformers**: https://github.com/huggingface/transformers
- **This Dockerfile**: Open an issue in this repository
