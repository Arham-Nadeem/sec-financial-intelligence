#Load project configuration and apply safe local environment overrides.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)

    #Keep personal SEC contact information out of committed configuration when desired.
    sec_user_agent = os.getenv("SEC_USER_AGENT")
    if sec_user_agent:
        settings["sec"]["user_agent"] = sec_user_agent
    return settings
