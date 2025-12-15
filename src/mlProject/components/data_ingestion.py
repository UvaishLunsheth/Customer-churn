import pandas as pd
from mlProject.entity.config_entity import DataIngestionConfig
from mlProject import logger
from mlProject.utils.common import create_directories
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> Path:
        """
        Downloads the dataset from a remote URL and saves it to the artifacts directory.
        Returns the path to the saved data file.
        """
        logger.info("Starting data ingestion process")

        # Create root directory for data ingestion
        create_directories([self.config.root_dir])

        try:
            logger.info(f"Reading dataset from URL: {self.config.source_URL}")

            # Read CSV directly from GitHub raw URL
            df = pd.read_csv(self.config.source_URL)

            logger.info("Dataset successfully downloaded")

            # Save dataset to artifacts directory
            df.to_csv(self.config.local_data_file, index=False)

            logger.info(
                f"Dataset saved to artifacts at: {self.config.local_data_file}"
            )

            return self.config.local_data_file

        except Exception as e:
            logger.error("Error occurred during data ingestion")
            raise e
