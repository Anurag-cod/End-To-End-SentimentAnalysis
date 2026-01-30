import os
import sys
import pandas as pd
from zipfile import ZipFile
from hate.logger import logging
from hate.exception import CustomException
from hate.configuration.gcloud_syncer import GCloudSync
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts
from hate.constants import COLUMN_HEADERS


class DataIngestion:
    """Data ingestion aligned with twitter-sentiment-analysis-with-lstm notebook flow."""

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = GCloudSync()

    def get_data_from_gcloud(self) -> None:
        try:
            logging.info("Entered the get_data_from_gcloud method of Data ingestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(
                self.data_ingestion_config.BUCKET_NAME,
                self.data_ingestion_config.ZIP_FILE_NAME,
                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,
            )

            logging.info("Exited the get_data_from_gcloud method of Data ingestion class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def unzip_and_clean(self):
        """Extract zip; notebook expects twitter_training.csv and twitter_validation.csv."""
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            training_path = self.data_ingestion_config.TRAINING_FILE_PATH
            validation_path = self.data_ingestion_config.VALIDATION_FILE_PATH

            # If extracted files have no headers, add headers (notebook: columns = ['Header1', 'company', 'labels', 'text'])
            for path, name in [(training_path, "training"), (validation_path, "validation")]:
                if os.path.isfile(path):
                    df = pd.read_csv(path, header=None)
                    if df.shape[1] == 4:
                        df.columns = COLUMN_HEADERS
                        df.to_csv(path, index=False)

            logging.info("Exited the unzip_and_clean method of Data ingestion class")
            return training_path, validation_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")

        try:
            self.get_data_from_gcloud()
            logging.info("Fetched the data from gcloud bucket")
            training_file_path, validation_file_path = self.unzip_and_clean()
            logging.info("Unzipped and prepared training and validation files")

            data_ingestion_artifacts = DataIngestionArtifacts(
                training_file_path=training_file_path,
                validation_file_path=validation_file_path,
            )

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
