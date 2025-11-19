"""
Data Slicer für Spark4
Teilt JSONL Dateien in Train/Validation Splits
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random


class DataSlicer:
    """Einfacher Train/Val Splitter für JSONL Dateien"""

    def __init__(self, train_ratio: float = 0.8, seed: int = 42):
        """
        Args:
            train_ratio: Anteil für Training (default: 0.8 = 80%)
            seed: Random Seed für Reproduzierbarkeit
        """
        self.train_ratio = train_ratio
        self.seed = seed
        random.seed(seed)

    def load_jsonl(self, input_path: Path) -> List[Dict[str, Any]]:
        """
        Lädt JSONL Datei

        Args:
            input_path: Pfad zur JSONL Datei

        Returns:
            Liste von JSON Objekten
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input Datei nicht gefunden: {input_path}")

        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        return data

    def save_jsonl(self, data: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Speichert Daten als JSONL

        Args:
            data: Liste von JSON Objekten
            output_path: Ausgabe Pfad
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def split(
        self,
        input_path: Path,
        output_dir: Path,
        train_name: str = "train.jsonl",
        val_name: str = "validation.jsonl",
        shuffle: bool = True
    ) -> Tuple[int, int]:
        """
        Splittet JSONL in Train/Val

        Args:
            input_path: Input JSONL Datei
            output_dir: Output Verzeichnis
            train_name: Name für Train Datei
            val_name: Name für Validation Datei
            shuffle: Daten shufflen vor Split

        Returns:
            Tuple (train_count, val_count)
        """
        # Daten laden
        data = self.load_jsonl(input_path)
        total = len(data)

        if total == 0:
            raise ValueError("Input Datei ist leer!")

        # Shufflen falls gewünscht
        if shuffle:
            random.shuffle(data)

        # Split berechnen
        train_size = int(total * self.train_ratio)

        train_data = data[:train_size]
        val_data = data[train_size:]

        # Speichern
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / train_name
        val_path = output_dir / val_name

        self.save_jsonl(train_data, train_path)
        self.save_jsonl(val_data, val_path)

        print(f"✓ Split completed:")
        print(f"  Total: {total} Zeilen")
        print(f"  Train: {len(train_data)} → {train_path}")
        print(f"  Val:   {len(val_data)} → {val_path}")

        return len(train_data), len(val_data)


def quick_split(
    input_path: str,
    output_dir: str = "data_warehouse/splits",
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[int, int]:
    """
    Schneller Helper für einfache Splits

    Args:
        input_path: Input JSONL Datei
        output_dir: Output Verzeichnis (default: data_warehouse/splits)
        train_ratio: Train Ratio (default: 0.8)
        seed: Random Seed (default: 42)

    Returns:
        Tuple (train_count, val_count)
    """
    slicer = DataSlicer(train_ratio=train_ratio, seed=seed)
    return slicer.split(
        input_path=Path(input_path),
        output_dir=Path(output_dir)
    )
