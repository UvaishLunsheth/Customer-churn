from pathlib import Path
import json
import numpy as np
import joblib

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from mlProject import logger
from mlProject.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _load_artifacts(self):
        model = joblib.load(self.config.model_path)
        X_test = np.load(self.config.test_data_path)
        y_test = np.load(self.config.y_test)

        return model, X_test, y_test

    def initiate_model_evaluation(self):
        logger.info("Starting model evaluation")

        model, X_test, y_test = self._load_artifacts()

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        output_path = Path(self.config.metrics_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Model evaluation metrics saved at: {output_path}")
        logger.info(f"Evaluation Metrics: {metrics}")

        return metrics
