# Minimal LLaMA Training

Ein einfaches Skript zum Trainieren von LLaMA mit LoRA-Adaptern.

## Installation

```bash
pip install -r requirements.txt
```

## Verwendung

```bash
python train_minimal.py
```

## Was macht das Skript?

1. **Lädt TinyLlama-1.1B** (kleinste verfügbare LLaMA-Variante)
2. **Erstellt ein Dummy-Dataset** mit 10 Textzeilen
3. **Trainiert für 1 Epoche** mit LoRA-Adaptern
4. **Speichert den Adapter** in `./lora_adapter/`

## Ausgabe

- `./output/` - Training-Checkpoints
- `./lora_adapter/` - Finaler LoRA-Adapter

## Features

- ✅ Keine komplizierten Abstraktionen
- ✅ Keine Registry oder Module
- ✅ Einfach und funktionsfähig
- ✅ GPU-Support (automatisch wenn verfügbar)
- ✅ Minimale Abhängigkeiten
