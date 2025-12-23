from dataclasses import dataclass
from pathlib import Path


# ================================
# Data Ingestion Config
# ================================

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_data_path: Path
    local_data_file: Path


# ================================
# Data Validation Config
# ================================

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    schema_raw: Path
    schema_processed: Path
    validation_report: Path


# ================================
# Data Cleaning Config
# ================================

@dataclass(frozen=True)
class DataCleaningConfig:
    root_dir: Path
    cleaned_data_path: Path


# ================================
# Feature Engineering Config
# ================================

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    featured_data_path: Path


# ================================
# Data Transformation Config
# ================================

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    transformed_train: Path
    transformed_test: Path
    y_train: Path
    y_test: Path
    preprocessor_path: Path


# ================================
# Model Trainer Config
# ================================

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_path: Path
    train_data_path: Path
    test_data_path: Path
    y_train: Path
    y_test: Path


# ================================
# Model Evaluation Config
# ================================

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    test_data_path: Path
    y_test: Path
    metrics_path: Path
