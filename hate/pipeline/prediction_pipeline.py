"""
Prediction pipeline aligned with twitter-sentiment-analysis-with-lstm notebook flow:
- process_text (lemmatization, stopwords, etc.)
- Tokenizer + pad_sequences(maxlen=100)
- 4-class model (Positive, Negative, Neutral, Irrelevant)
"""
import os
import sys
import pickle
import keras
from keras.utils import pad_sequences

from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from hate.configuration.gcloud_syncer import GCloudSync
from hate.components.data_transforamation import process_text


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.gcloud = GCloudSync()

    def get_model_from_gcloud(self) -> str:
        logging.info("Entered the get_model_from_gcloud method of PredictionPipeline class")
        try:
            os.makedirs(self.model_path, exist_ok=True)
            self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, self.model_path)
            best_model_path = os.path.join(self.model_path, self.model_name)
            logging.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_tokenizer_and_label_encoder(self, model_dir: str):
        tokenizer_path = os.path.join(model_dir, TOKENIZER_FILE_NAME)
        label_encoder_path = os.path.join(model_dir, LABEL_ENCODER_FILE_NAME)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(label_encoder_path, 'rb') as handle:
            label_encoder = pickle.load(handle)
        return tokenizer, label_encoder

    def predict(self, text: str, best_model_path: str = None):
        """
        Clean text with process_text, tokenize, pad (maxlen=100), predict 4-class sentiment.
        Returns class name: Positive, Negative, Neutral, or Irrelevant.
        """
        logging.info("Running the predict function")
        try:
            if best_model_path is None:
                best_model_path = self.get_model_from_gcloud()

            model_dir = os.path.dirname(best_model_path)
            load_model = keras.models.load_model(best_model_path)
            tokenizer, label_encoder = self._load_tokenizer_and_label_encoder(model_dir)

            cleaned = process_text(text)
            text_list = [cleaned]
            seq = tokenizer.texts_to_sequences(text_list)
            padded = pad_sequences(seq, maxlen=MAX_LEN, truncating='post')

            pred = load_model.predict(padded)
            pred_class = pred.argmax(axis=1)[0]
            label_name = label_encoder.inverse_transform([pred_class])[0]
            logging.info(f"Prediction: {label_name}")
            return label_name
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, text: str):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            best_model_path = self.get_model_from_gcloud()
            predicted_label = self.predict(text, best_model_path=best_model_path)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_label
        except Exception as e:
            raise CustomException(e, sys) from e
