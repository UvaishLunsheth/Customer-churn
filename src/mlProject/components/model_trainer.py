from pathlib import Path
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.utils.common import read_yaml
from mlProject.constants import PARAMS_FILE_PATH


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)

    def _load_data(self):
        X_train = np.load(self.config.train_data_path)
        X_test = np.load(self.config.test_data_path)
        y_train = np.load(self.config.y_train)
        y_test = np.load(self.config.y_test)

        return X_train, X_test, y_train, y_test

    def initiate_model_training(self):
        logger.info("Starting model training")

        X_train, X_test, y_train, y_test = self._load_data()

        lr_params = self.params["logistic_regression"]

        model = LogisticRegression(
            max_iter=lr_params["max_iter"],
            class_weight=lr_params["class_weight"],
            solver=lr_params["solver"]
        )

        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"Model training completed. ROC-AUC: {roc_auc:.4f}")

        output_path = Path(self.config.model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, output_path)

        logger.info(f"Trained model saved at: {output_path}")

        return output_path, roc_auc
