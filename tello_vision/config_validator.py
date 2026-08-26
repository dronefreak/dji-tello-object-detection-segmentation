"""Configuration validation utilities.

Provides a small, dependency-free schema check for ``config.yaml`` so
that missing or malformed configuration surfaces as an actionable error
message at startup instead of a raw ``KeyError`` deep inside the
application (e.g. while initializing the detector or drone controller).
"""

from typing import Any, Dict, List

# Top-level sections every config.yaml must define.
REQUIRED_TOP_LEVEL_SECTIONS = [
    "detector",
    "drone",
    "processing",
    "visualization",
    "controls",
]

SUPPORTED_DETECTOR_BACKENDS = ["yolov8", "detectron2"]


class ConfigError(ValueError):
    """Raised when the application configuration is invalid or incomplete."""


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the structure of a loaded ``config.yaml`` dictionary.

    Args:
        config: Parsed configuration dictionary.

    Raises:
        ConfigError: If required sections or keys are missing or invalid.
    """
    if not isinstance(config, dict):
        raise ConfigError(
            "Configuration file must contain a top-level mapping "
            f"(got {type(config).__name__})."
        )

    missing = _missing_keys(config, REQUIRED_TOP_LEVEL_SECTIONS)
    if missing:
        raise ConfigError(
            "Missing required configuration section(s): "
            f"{', '.join(missing)}. See config.yaml for the expected format."
        )

    _validate_detector_section(config["detector"])
    _validate_processing_section(config["processing"])


def _missing_keys(config: Dict[str, Any], keys: List[str]) -> List[str]:
    return [key for key in keys if key not in config]


def _validate_detector_section(detector_config: Any) -> None:
    if not isinstance(detector_config, dict):
        raise ConfigError("'detector' section must be a mapping.")

    if "backend" not in detector_config:
        raise ConfigError(
            "'detector.backend' is required "
            f"(supported: {', '.join(SUPPORTED_DETECTOR_BACKENDS)})."
        )

    backend = detector_config["backend"]
    if backend not in SUPPORTED_DETECTOR_BACKENDS:
        raise ConfigError(
            f"Unsupported detector backend '{backend}'. "
            f"Supported backends: {', '.join(SUPPORTED_DETECTOR_BACKENDS)}."
        )

    if backend not in detector_config:
        raise ConfigError(
            f"'detector.{backend}' configuration block is required when "
            f"'detector.backend' is '{backend}'."
        )


def _validate_processing_section(processing_config: Any) -> None:
    if not isinstance(processing_config, dict):
        raise ConfigError("'processing' section must be a mapping.")

    if "output_dir" not in processing_config:
        raise ConfigError("'processing.output_dir' is required.")
