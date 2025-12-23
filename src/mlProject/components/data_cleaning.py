from pathlib import Path
import pandas as pd
import numpy as np

from mlProject import logger
from mlProject.entity.config_entity import DataCleaningConfig


class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config

    def initiate_data_cleaning(self, raw_data_path: Path) -> Path:
        """
        Cleans raw data and saves cleaned dataset.

        Args:
            raw_data_path (Path): path to validated raw data

        Returns:
            Path: path to cleaned dataset
        """
        logger.info("Starting data cleaning process")

        df = pd.read_csv(raw_data_path)

        logger.info("Replacing blank strings with NaN")
        df.replace(" ", np.nan, inplace=True)

        logger.info("Dropping rows with missing TotalCharges")
        df = df.dropna(subset=["TotalCharges"])

        logger.info("Converting TotalCharges to float")
        df["TotalCharges"] = df["TotalCharges"].astype(float)

        logger.info("Dropping identifier column: customerID")
        df.drop(columns=["customerID"], inplace=True)

        output_path = Path(self.config.cleaned_data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        logger.info(f"Data cleaning completed. Cleaned data saved at: {output_path}")
        logger.info(f"Cleaned dataset shape: {df.shape}")

        return output_path
