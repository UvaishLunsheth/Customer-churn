import pandas as pd
from mlProject.entity.config_entity import DataValidationConfig
from mlProject import logger
from mlProject.utils.common import create_directories
from pathlib import Path


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validate whether all required columns are present in the dataset.
        """
        try:
            logger.info("Starting data validation")

            # Create validation directory
            create_directories([self.config.root_dir])

            # Load dataset
            data = pd.read_csv(
                Path("artifacts/data_ingestion/data.csv")
            )

            all_columns = list(data.columns)
            schema_columns = list(self.config.all_schema["columns"].keys())

            validation_status = True

            for col in schema_columns:
                if col not in all_columns:
                    logger.error(f"Missing column: {col}")
                    validation_status = False

            # Write validation status
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(str(validation_status))

            if validation_status:
                logger.info("Data validation passed")
            else:
                logger.warning("Data validation failed")

            return validation_status

        except Exception as e:
            logger.error("Error occurred during data validation")
            raise e
