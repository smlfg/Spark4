"""
Config Registry für Spark4
Lädt und validiert YAML Konfigurationen
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigRegistry:
    """Einfacher Config Loader für YAML Files"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: Pfad zur Config-Datei (optional)
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

        if config_path:
            self.load(config_path)

    def load(self, config_path: Path) -> Dict[str, Any]:
        """
        Lädt YAML Config aus File

        Args:
            config_path: Pfad zur YAML Datei

        Returns:
            Config Dictionary
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config nicht gefunden: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Holt Wert aus Config

        Args:
            key: Config Key (unterstützt nested keys mit Punkt: "model.name")
            default: Default Wert falls Key nicht existiert

        Returns:
            Config Wert
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Direkter Zugriff via registry['key']"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Config key nicht gefunden: {key}")
        return value

    def __contains__(self, key: str) -> bool:
        """Check ob Key existiert"""
        return self.get(key) is not None
