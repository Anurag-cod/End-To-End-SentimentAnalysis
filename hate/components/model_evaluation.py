"""
Model evaluation aligned with notebook flow: 4-class sentiment, tokenizer + label_encoder from trainer.
Compares current model vs best model from GCloud using accuracy (higher is better).
"""
import os
import sys
import pickle
import keras
import pandas as pd
from keras.utils import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from hate.configuration.gcloud_syncer import GCloudSync
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts=None,
    ):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.gcloud = GCloudSync()

    def get_best_model_from_gcloud(self) -> str:
        try:
            logging.info("Entered the get_best_model_from_gcloud method of Model Evaluation class")
            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)
            self.gcloud.sync_folder_from_gcloud(
                self.model_evaluation_config.BUCKET_NAME,
                self.model_evaluation_config.MODEL_NAME,
                self.model_evaluation_config.BEST_MODEL_DIR_PATH,
            )
            best_model_path = os.path.join(
                self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                self.model_evaluation_config.MODEL_NAME,
            )
            logging.info("Exited the get_best_model_from_gcloud method of Model Evaluation class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_test_data_and_prepare(self):
        """Load x_test, y_test from CSV; load tokenizer and label_encoder; return padded X and encoded y."""
        x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col=0)
        y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)
        x_test = x_test.squeeze().astype(str)
        y_test = y_test.squeeze()

        tokenizer_path = os.path.join(
            os.path.dirname(self.model_trainer_artifacts.trained_model_path),
            TOKENIZER_FILE_NAME,
        )
        label_encoder_path = os.path.join(
            os.path.dirname(self.model_trainer_artifacts.trained_model_path),
            LABEL_ENCODER_FILE_NAME,
        )
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(label_encoder_path, 'rb') as handle:
            label_encoder = pickle.load(handle)

        test_sequences = tokenizer.texts_to_sequences(x_test)
        test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN, truncating='post')
        y_test_encoded = label_encoder.transform(y_test)
        y_test_cat = to_categorical(y_test_encoded, num_classes=NUM_CLASSES)
        return test_sequences_matrix, y_test_cat, y_test_encoded, tokenizer, label_encoder

    def evaluate(self, model_path: str):
        """
        Evaluate model at model_path on stored x_test/y_test.
        Returns [loss, accuracy]; we use accuracy (index 1) for comparison (higher is better).
        """
        try:
            logging.info("Entering the evaluate function of Model Evaluation class")
            X_test, y_test_cat, y_test_encoded, _, _ = self._load_test_data_and_prepare()

            load_model = keras.models.load_model(model_path)
            result = load_model.evaluate(X_test, y_test_cat)
            # result is [loss, accuracy] for metrics=['accuracy']
            logging.info(f"Model evaluation result (loss, accuracy): {result}")

            predictions = load_model.predict(X_test)
            pred_classes = predictions.argmax(axis=1)
            cm = confusion_matrix(y_test_encoded, pred_classes)
            logging.info(f"Confusion matrix: {cm}")
            return result
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Compare currently trained model with best model from GCloud.
        Use accuracy (index 1): higher is better; accept trained model if its accuracy >= best.
        """
        logging.info("Initiate Model Evaluation")
        try:
            trained_model_path = self.model_trainer_artifacts.trained_model_path
            trained_result = self.evaluate(trained_model_path)
            trained_accuracy = trained_result[1]

            best_model_path = self.get_best_model_from_gcloud()

            if not os.path.isfile(best_model_path):
                is_model_accepted = True
                logging.info("Best model not present in GCloud; accepting trained model")
            else:
                best_result = self.evaluate(best_model_path)
                best_accuracy = best_result[1]
                # Higher accuracy is better
                is_model_accepted = trained_accuracy >= best_accuracy
                logging.info(
                    f"Trained accuracy: {trained_accuracy}, Best accuracy: {best_accuracy}, "
                    f"Accepted: {is_model_accepted}"
                )

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
