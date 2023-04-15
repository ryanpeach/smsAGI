from typing import Optional
import yaml
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config/default.yaml"
CONFIG_MEMO = {
    DEFAULT_CONFIG: yaml.safe_load(DEFAULT_CONFIG.read_text())
}

class Config:

    def __init__(self, config: Optional[Path] = None) -> None:
        self.config_file = config
        if config is not None:
            if config not in CONFIG_MEMO:
                CONFIG_MEMO[config] = yaml.safe_load(config.read_text())
            self.config = CONFIG_MEMO[config]
        else:
            self.config = {}
        self.default_config = CONFIG_MEMO[DEFAULT_CONFIG]
