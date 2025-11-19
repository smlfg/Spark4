# Spark4 - LLaMA Training mit Unsloth

Produktionsreifes Training-Framework für LLaMA mit Unsloth-Optimierungen.

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Dummy-Daten generieren

```bash
python generate_dummy_data.py
```

Erstellt `dummy.jsonl` mit 50 Test-Samples (instruction/input/output Format).

### 3. Training starten

```bash
python train.py
```

**Produktions-Training mit Unsloth:**
- Lädt `unsloth/tinyllama-bnb-2b` (schnelles Test-Modell)
- Trainiert mit Daten aus `dummy.jsonl`
- 10 Training-Steps (konfigurierbar)
- Speichert LoRA-Adapter in `outputs/`

### 4. (Optional) Minimales Basis-Skript

```bash
python train_minimal.py
```

Einfaches Referenz-Skript ohne Unsloth (für Vergleich).

## 📁 Projektstruktur

```
Spark4/
├── train.py                  # Haupt-Training (Unsloth-optimiert)
├── train_minimal.py          # Basis-Version (Referenz)
├── generate_dummy_data.py    # Generiert Test-Daten
├── dummy.jsonl              # Training-Daten (50 Samples)
├── outputs/                  # Gespeicherte Adapter
└── requirements.txt          # Dependencies
```

## ⚙️ train.py Features

- ✅ **Unsloth-Optimierung** - 2x schneller, 50% weniger VRAM
- ✅ **4-bit Quantization** - Läuft auf Consumer-GPUs
- ✅ **Error Handling** - Try/Catch um Training Loop
- ✅ **JSONL Data Loading** - Flexibles Datenformat
- ✅ **LoRA Adapter** - Effiziente Fine-Tuning Methode
- ✅ **Correct Import Order** - Unsloth MUSS zuerst importiert werden

## 🛠️ Konfiguration (train.py)

Hardcoded Parameters (für schnelle Tests):
- Model: `unsloth/tinyllama-bnb-2b`
- Max Steps: `10`
- Batch Size: `1`
- LoRA Rank: `16`
- Max Sequence Length: `512`

## 📊 Ausgabe

Nach Training:
- `outputs/adapter_config.json` - LoRA Konfiguration
- `outputs/adapter_model.safetensors` - Trainierte Weights
- `outputs/tokenizer_config.json` - Tokenizer Settings

## 🔧 Troubleshooting

**"dummy.jsonl not found"**
```bash
python generate_dummy_data.py
```

**Import Error mit Unsloth**
- Stelle sicher, dass `from unsloth import FastLanguageModel` die ERSTE Zeile ist
- Reinstall: `pip install --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`

**CUDA Out of Memory**
- Reduziere `MAX_SEQ_LENGTH` in train.py
- Verwende `load_in_4bit=True` (bereits aktiviert)
