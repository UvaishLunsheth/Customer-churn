from mlProject.pipeline.pipeline import TrainingPipeline
from mlProject import logger

STAGE_NAME = "Training Pipeline"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.run()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e