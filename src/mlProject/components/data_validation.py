import json
from pathlib import Path

import pandas as pd

from mlProject import logger
from mlProject.entity.config_entity import DataValidationConfig
from mlProject.utils.common import read_yaml


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def _load_schema(self):
        return read_yaml(self.config.schema_raw)

    def _validate_columns(self, df: pd.DataFrame, schema: dict) -> bool:
        schema_columns = schema["columns"].keys()
        df_columns = df.columns

        missing_columns = set(schema_columns) - set(df_columns)
        extra_columns = set(df_columns) - set(schema_columns)

        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False

        if extra_columns:
            logger.error(f"Unexpected columns: {extra_columns}")
            return False

        return True

    def _validate_column_count(self, df: pd.DataFrame, schema: dict) -> bool:
        expected_columns = schema["dataset"]["shape"]["columns"]
        return df.shape[1] == expected_columns

    def _validate_dtypes(self, df: pd.DataFrame, schema: dict) -> bool:
        schema_dtypes = schema["columns"]

        for col, expected_dtype in schema_dtypes.items():
            if col not in df.columns:
                continue

            actual_dtype = str(df[col].dtype)

            if expected_dtype not in actual_dtype:
                logger.warning(
                    f"Data type mismatch in column '{col}': "
                    f"expected {expected_dtype}, got {actual_dtype}"
                )

        return True

    def initiate_data_validation(self, data_path: Path) -> bool:
        logger.info("Starting data validation")

        df = pd.read_csv(data_path)
        schema = self._load_schema()

        validation_status = {
            "column_names_valid": self._validate_columns(df, schema),
            "column_count_valid": self._validate_column_count(df, schema),
            "dtypes_checked": self._validate_dtypes(df, schema)
        }

        with open(self.config.validation_report, "w") as f:
            json.dump(validation_status, f, indent=4)

        logger.info(f"Validation report saved to: {self.config.validation_report}")

        if not all(validation_status.values()):
            raise ValueError("Data validation failed. Check validation report.")

        logger.info("Data validation successful")
        return True
