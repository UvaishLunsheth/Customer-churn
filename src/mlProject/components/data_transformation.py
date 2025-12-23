from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig
from mlProject.utils.common import read_yaml
from mlProject.constants import PARAMS_FILE_PATH, SCHEMA_PROCESSED_FILE_PATH


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.schema = read_yaml(SCHEMA_PROCESSED_FILE_PATH)

    def _get_preprocessor(self):
        # 1. Get column lists from schema
        numerical_cols = self.schema["numerical_columns"]
        
        # Define the specific columns as per your requirements
        nominal_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 
            'PaymentMethod', 'HasInternet'
        ]

        ordinal_cols = ['Contract', 'TenureGroup', 'MonthlyChargeLevel', 'SupportRisk', 'ContractRisk']

        # 2. Define Ordinal Hierarchies
        # These lists ensure the OrdinalEncoder assigns 0, 1, 2 in the correct order
        contract_categories = ['Month-to-month', 'One year', 'Two year']
        tenure_categories = ['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years']
        charge_categories = ['Low', 'Medium', 'High']
        support_risk_categories = ['LowRisk', 'HighRisk']
        contract_risk_categories = ['Low', 'Medium', 'High']

        # 3. Create Scaler and Encoders
        scaler_type = self.params["data_transformation"]["scaling"]["numerical_scaler"]
        scaler = StandardScaler() if scaler_type == "standard" else "passthrough"

        nominal_transformer = OneHotEncoder(
            drop="first" if self.params["data_transformation"]["encoding"]["drop_first"] else None,
            handle_unknown="ignore",
            sparse_output=False # Updated from 'sparse' for newer sklearn versions
        )

        ordinal_transformer = OrdinalEncoder(categories=[
            contract_categories,
            tenure_categories,
            charge_categories,
            support_risk_categories,
            contract_risk_categories
        ])

        # 4. Build the final ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", scaler, numerical_cols),
                ("nominal", nominal_transformer, nominal_cols),
                ("ordinal", ordinal_transformer, ordinal_cols)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, featured_data_path: Path):
        logger.info("Starting data transformation")

        df = pd.read_csv(featured_data_path)

        target_col = self.schema["target_column"]
        X = df.drop(columns=[target_col])
        y = df[target_col].map({"Yes": 1, "No": 0})

        split_cfg = self.params["data_split"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=split_cfg["test_size"],
            random_state=self.params["general"]["random_state"],
            stratify=y if split_cfg["stratify"] else None
        )

        preprocessor = self._get_preprocessor()

        logger.info("Fitting and applying preprocessor on training data")
        # Use fit_transform on train and transform on test to avoid data leakage
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Save the preprocessor and the numpy arrays
        joblib.dump(preprocessor, self.config.preprocessor_path)

        np.save(self.config.transformed_train, X_train_transformed)
        np.save(self.config.transformed_test, X_test_transformed)
        np.save(self.config.y_train, y_train.values)
        np.save(self.config.y_test, y_test.values)

        logger.info(f"Data transformation completed. Preprocessor saved at: {self.config.preprocessor_path}")

        return (
            self.config.transformed_train,
            self.config.transformed_test,
            self.config.y_train,
            self.config.y_test
        )