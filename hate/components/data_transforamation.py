"""
Data transformation aligned with twitter-sentiment-analysis-with-lstm notebook flow:
- Load training + validation, add headers, drop unneeded columns
- Concat, dropna, drop_duplicates
- process_text: regex, lower, word_tokenize, WordNetLemmatizer, stopwords, len>3, unique order
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


def process_text(text: str) -> str:
    """
    Notebook-aligned text cleaning: regex, lower, tokenize, lemmatize, stopwords, len>3, unique order.
    Returns space-joined cleaned words for storage and tokenizer.
    """
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 3]
    indices = np.unique(words, return_index=True)[1]
    cleaned = np.array(words)[np.sort(indices)].tolist()
    return " ".join(cleaned)


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifacts: DataIngestionArtifacts,
    ):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def load_and_prepare(self) -> pd.DataFrame:
        """Load training and validation CSVs, set headers, drop columns, concat (notebook flow)."""
        try:
            logging.info("Entered load_and_prepare (notebook: training + test, drop Header1 & company)")
            training_path = self.data_ingestion_artifacts.training_file_path
            validation_path = self.data_ingestion_artifacts.validation_file_path

            training = pd.read_csv(training_path)
            test = pd.read_csv(validation_path)

            if 'Header1' not in training.columns and training.shape[1] == 4:
                training.columns = self.data_transformation_config.COLUMN_HEADERS
                test.columns = self.data_transformation_config.COLUMN_HEADERS

            training.drop(columns=self.data_transformation_config.DROP_INGESTION_COLUMNS, inplace=True)
            test.drop(columns=self.data_transformation_config.DROP_INGESTION_COLUMNS, inplace=True)

            sentiment = pd.concat([training, test], ignore_index=True)
            sentiment.dropna(inplace=True)
            sentiment.drop_duplicates(inplace=True)

            logging.info(f"Combined and cleaned dataframe shape: {sentiment.shape}")
            return sentiment
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")

            df = self.load_and_prepare()
            text_col = self.data_transformation_config.TEXT_COLUMN
            labels_col = self.data_transformation_config.LABELS_COLUMN

            logging.info("Applying process_text (lemmatization, stopwords, etc.) on text column")
            df[text_col] = df[text_col].astype(str).apply(process_text)

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index=False, header=True)

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_data_path=self.data_transformation_config.TRANSFORMED_FILE_PATH,
            )
            logging.info("Returning the DataTransformationArtifacts")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
