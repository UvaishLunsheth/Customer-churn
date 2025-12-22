import yaml
from pathlib import Path

from mlProject.constants import CONFIG_FILE_PATH
from mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataCleaningConfig,
    FeatureEngineeringConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)

from mlProject.utils.common import create_directories


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
    ):
        with open(config_filepath, "r") as file:
            self.config = yaml.safe_load(file)

        self.artifacts_root = Path(self.config["artifacts_root"])
        create_directories([self.artifacts_root])

    # ================================
    # Data Ingestion
    # ================================

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]

        create_directories([config["root_dir"]])

        return DataIngestionConfig(
            root_dir=Path(config["root_dir"]),
            source_data_path=Path(config["source_data_path"]),
            local_data_file=Path(config["local_data_file"]),
        )

    # ================================
    # Data Validation
    # ================================

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config["data_validation"]

        create_directories([config["root_dir"]])

        return DataValidationConfig(
            root_dir=Path(config["root_dir"]),
            schema_raw=Path(config["schema_raw"]),
            schema_processed=Path(config["schema_processed"]),
            validation_report=Path(config["validation_report"]),
        )

    # ================================
    # Data Cleaning
    # ================================

    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config["data_cleaning"]

        create_directories([config["root_dir"]])

        return DataCleaningConfig(
            root_dir=Path(config["root_dir"]),
            cleaned_data_path=Path(config["cleaned_data_path"]),
        )

    # ================================
    # Feature Engineering
    # ================================

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        config = self.config["feature_engineering"]

        create_directories([config["root_dir"]])

        return FeatureEngineeringConfig(
            root_dir=Path(config["root_dir"]),
            featured_data_path=Path(config["featured_data_path"]),
        )

    # ================================
    # Data Transformation
    # ================================

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config["data_transformation"]

        create_directories([config["root_dir"]])

        return DataTransformationConfig(
            root_dir=Path(config["root_dir"]),
            transformed_train=Path(config["transformed_train"]),
            transformed_test=Path(config["transformed_test"]),
            y_train=Path(config["y_train"]),
            y_test=Path(config["y_test"]),
            preprocessor_path=Path(config["preprocessor_path"]),
        )

    # ================================
    # Model Trainer
    # ================================

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config["model_trainer"]

        create_directories([config["root_dir"]])

        return ModelTrainerConfig(
            root_dir=Path(config["root_dir"]),
            model_path=Path(config["model_path"]),
        )

    # ================================
    # Model Evaluation
    # ================================

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config["model_evaluation"]

        create_directories([config["root_dir"]])

        return ModelEvaluationConfig(
            root_dir=Path(config["root_dir"]),
            metrics_path=Path(config["metrics_path"]),
        )
