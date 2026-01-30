"""
Model trainer aligned with twitter-sentiment-analysis-with-lstm notebook flow:
- Read transformed CSV (text, labels)
- train_test_split(test_size=0.2)
- Tokenizer(num_words=20000), fit_on_texts(X_train), pad_sequences(maxlen=100, truncating='post')
- LabelEncoder for 4 classes, to_categorical
- Bidirectional LSTM (4-class softmax), Adam(0.0001), categorical_crossentropy
- Save model, tokenizer, label_encoder
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
# from keras.utils import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from hate.ml.model import ModelArchitecture


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifacts: DataTransformationArtifacts,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def split_data(self, csv_path: str):
        """Notebook: x = text, y = labels; train_test_split(test_size=0.2, random_state=42)."""
        try:
            logging.info("Entered the split_data function")
            df = pd.read_csv(csv_path, index_col=False)
            text_col = self.model_trainer_config.TEXT_COLUMN
            labels_col = self.model_trainer_config.LABELS_COLUMN

            x = df[text_col].astype(str)
            y = df[labels_col]

            x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                test_size=self.model_trainer_config.TEST_SIZE,
                random_state=self.model_trainer_config.RANDOM_STATE,
            )
            logging.info("Exited the split_data function")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys) from e

    def tokenize_and_pad(self, x_train, x_test):
        """Notebook: Tokenizer(num_words=20000), fit_on_texts(X_train), pad_sequences(maxlen=100, truncating='post')."""
        try:
            logging.info("Applying tokenization and padding (notebook: max_vocab=20000, maxlen=100)")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)

            x_train_seq = tokenizer.texts_to_sequences(x_train)
            x_test_seq = tokenizer.texts_to_sequences(x_test)

            x_train_pad = pad_sequences(
                x_train_seq,
                maxlen=self.model_trainer_config.MAX_LEN,
                truncating='post',
            )
            x_test_pad = pad_sequences(
                x_test_seq,
                maxlen=self.model_trainer_config.MAX_LEN,
                truncating='post',
            )

            vocab_size = len(tokenizer.word_index) + 1
            logging.info(f"Vocab size (for Embedding): {vocab_size}")
            return x_train_pad, x_test_pad, tokenizer, vocab_size
        except Exception as e:
            raise CustomException(e, sys) from e

    def encode_labels(self, y_train, y_test):
        """Notebook: LabelEncoder for 4 classes (Negative, Positive, Neutral, Irrelevant), then to_categorical."""
        try:
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            y_train_cat = to_categorical(y_train_encoded, num_classes=self.model_trainer_config.NUM_CLASSES)
            y_test_cat = to_categorical(y_test_encoded, num_classes=self.model_trainer_config.NUM_CLASSES)
            return y_train_cat, y_test_cat, label_encoder
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            x_train, x_test, y_train, y_test = self.split_data(
                csv_path=self.data_transformation_artifacts.transformed_data_path,
            )

            x_train_pad, x_test_pad, tokenizer, vocab_size = self.tokenize_and_pad(x_train, x_test)
            y_train_cat, y_test_cat, label_encoder = self.encode_labels(y_train, y_test)

            model_architecture = ModelArchitecture(config=self.model_trainer_config)
            model = model_architecture.get_model(vocab_size=vocab_size)

            logging.info("Starting model training (notebook: 20 epochs)")
            model.fit(
                x_train_pad,
                y_train_cat,
                batch_size=self.model_trainer_config.BATCH_SIZE,
                epochs=self.model_trainer_config.EPOCH,
                validation_split=self.model_trainer_config.VALIDATION_SPLIT,
            )
            logging.info("Model training finished")

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)

            tokenizer_path = os.path.join(
                self.model_trainer_config.TRAINED_MODEL_DIR,
                self.model_trainer_config.TOKENIZER_FILE_NAME,
            )
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            label_encoder_path = os.path.join(
                self.model_trainer_config.TRAINED_MODEL_DIR,
                self.model_trainer_config.LABEL_ENCODER_FILE_NAME,
            )
            with open(label_encoder_path, 'wb') as handle:
                pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)
            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH,
            )
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
