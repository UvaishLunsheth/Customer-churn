from pathlib import Path
import pandas as pd
import numpy as np

from mlProject import logger
from mlProject.entity.config_entity import FeatureEngineeringConfig


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    @staticmethod
    def _tenure_group(tenure: int) -> str:
        if tenure <= 12:
            return "0-1 Year"
        elif tenure <= 24:
            return "1-2 Years"
        elif tenure <= 48:
            return "2-4 Years"
        else:
            return "4+ Years"

    def initiate_feature_engineering(self, cleaned_data_path: Path) -> Path:
        """
        Applies feature engineering and saves featured dataset.

        Args:
            cleaned_data_path (Path): path to cleaned dataset

        Returns:
            Path: path to feature-engineered dataset
        """
        logger.info("Starting feature engineering process")

        df = pd.read_csv(cleaned_data_path)

        logger.info("Creating TenureGroup feature")
        df["TenureGroup"] = df["tenure"].apply(self._tenure_group)

        logger.info("Creating MonthlyChargeLevel feature")
        df["MonthlyChargeLevel"] = pd.qcut(
            df["MonthlyCharges"],
            q=3,
            labels=["Low", "Medium", "High"]
        )

        service_cols = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        logger.info("Creating TotalServices feature")
        df["TotalServices"] = df[service_cols].apply(
            lambda x: sum(x == "Yes"), axis=1
        )

        logger.info("Creating HasInternet feature")
        df["HasInternet"] = df["InternetService"].apply(
            lambda x: "No" if x == "No" else "Yes"
        )

        logger.info("Creating SupportRisk feature")
        df["SupportRisk"] = df[["OnlineSecurity", "TechSupport"]].apply(
            lambda x: "HighRisk" if all(x == "No") else "LowRisk",
            axis=1
        )

        logger.info("Creating ContractRisk feature")
        df["ContractRisk"] = df["Contract"].map({
            "Month-to-month": "High",
            "One year": "Medium",
            "Two year": "Low"
        })

        logger.info("Creating AvgMonthlySpend feature")
        df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"]
        df["AvgMonthlySpend"].replace([np.inf, -np.inf], 0, inplace=True)

        output_path = Path(self.config.featured_data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        logger.info(f"Feature engineering completed. Data saved at: {output_path}")
        logger.info(f"Featured dataset shape: {df.shape}")

        return output_path
