import shutil
from pathlib import Path

from mlProject import logger
from mlProject.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> Path:
        """
        Copies raw data from source to artifacts directory.

        Returns:
            Path: path to ingested raw data file
        """
        logger.info("Starting data ingestion process")

        source_path = Path(self.config.source_data_path)
        destination_path = Path(self.config.local_data_file)

        if not source_path.exists():
            raise FileNotFoundError(f"Source data not found at: {source_path}")

        shutil.copy(source_path, destination_path)

        logger.info(f"Data ingestion completed. File saved at: {destination_path}")

        return destination_path
