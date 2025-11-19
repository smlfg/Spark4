# Spark4

Saubere Portierung von Spark3 mit gefixten Strukturfehlern.

## Ordnerstruktur

```
Spark4/
├── spark4_core/          # Core Funktionen (NICHT "core" wegen Konflikten)
│   ├── __init__.py
│   ├── registry.py       # YAML Config Loader
│   └── slicer.py         # JSONL Train/Val Splitter
├── data_warehouse/       # Daten (NICHT "datasets"!)
├── model_registry/       # Modelle (NICHT "models"!)
├── outputs/              # Training Outputs
├── config_template.yaml  # Template für Training Config
└── gpu_check.py          # CUDA Check Script
```

## Wichtige Design Decisions

### Vermiedene Namenskonflikte
- **`spark4_core/`** statt `core` (vermeidet Python core conflicts)
- **`data_warehouse/`** statt `datasets` (HuggingFace Konflikt)
- **`model_registry/`** statt `models` (Library Konflikt)
- Keine Namen wie `tokenizers`, `transformers` etc.

## Quick Start

### 1. Config erstellen
```bash
cp config_template.yaml my_config.yaml
# Anpassen nach Bedarf
```

### 2. Config laden
```python
from spark4_core.registry import ConfigRegistry

registry = ConfigRegistry("my_config.yaml")
print(registry.get("model.name"))  # "gpt2"
```

### 3. Daten splitten
```python
from spark4_core.slicer import quick_split

# Einfacher 80/20 Split
quick_split(
    input_path="data_warehouse/raw/data.jsonl",
    output_dir="data_warehouse/splits"
)
```

Oder detaillierter:
```python
from spark4_core.slicer import DataSlicer

slicer = DataSlicer(train_ratio=0.8, seed=42)
train_count, val_count = slicer.split(
    input_path="data.jsonl",
    output_dir="data_warehouse/splits",
    train_name="train.jsonl",
    val_name="validation.jsonl"
)
```

### 4. GPU Check
```bash
python gpu_check.py
```

## Docker-Ready

Dieses Projekt ist für Docker-Container optimiert.
Koordination mit Infra-Agent läuft parallel.

## Architektur Prinzipien

1. **Einfachheit** - Keine unnötige Komplexität
2. **Saubere Struktur** - Keine Library-Konflikte
3. **Klarheit** - Selbsterklärender Code
4. **Lessons Learned** - Spark3 Fehler vermieden

## Version

**4.0.0** - Komplette Neustrukturierung
