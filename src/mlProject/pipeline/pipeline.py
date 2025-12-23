from mlProject import logger
from mlProject.config.configuration import ConfigurationManager

from mlProject.components.data_ingestion import DataIngestion
from mlProject.components.data_validation import DataValidation
from mlProject.components.data_cleaning import DataCleaning
from mlProject.components.feature_engineering import FeatureEngineering
from mlProject.components.data_transformation import DataTransformation
from mlProject.components.model_trainer import ModelTrainer
from mlProject.components.model_evaluation import ModelEvaluation


class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self):
        try:
            logger.info("===== Starting Training Pipeline =====")

            # Data Ingestion
            ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(ingestion_config)
            raw_data_path = data_ingestion.initiate_data_ingestion()

            # Data Validation
            validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(validation_config)
            data_validation.initiate_data_validation(raw_data_path)

            # Data Cleaning
            cleaning_config = self.config_manager.get_data_cleaning_config()
            data_cleaning = DataCleaning(cleaning_config)
            cleaned_data_path = data_cleaning.initiate_data_cleaning(raw_data_path)

            # Feature Engineering
            fe_config = self.config_manager.get_feature_engineering_config()
            feature_engineering = FeatureEngineering(fe_config)
            featured_data_path = feature_engineering.initiate_feature_engineering(
                cleaned_data_path
            )

            # Data Transformation
            transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(transformation_config)
            data_transformation.initiate_data_transformation(featured_data_path)

            # Model Training
            trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(trainer_config)
            model_path, roc_auc = model_trainer.initiate_model_training()

            # Model Evaluation
            evaluation_config = self.config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(evaluation_config)
            metrics = model_evaluation.initiate_model_evaluation()

            logger.info("===== Training Pipeline Completed Successfully =====")
            logger.info(f"Final ROC-AUC: {metrics['roc_auc']:.4f}")

        except Exception as e:
            logger.exception("Pipeline failed")
            raise e
