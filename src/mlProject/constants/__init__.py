from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "params.yaml"

SCHEMA_RAW_FILE_PATH = PROJECT_ROOT / "schema.yaml"
SCHEMA_PROCESSED_FILE_PATH = PROJECT_ROOT  / "schema_processed.yaml"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
