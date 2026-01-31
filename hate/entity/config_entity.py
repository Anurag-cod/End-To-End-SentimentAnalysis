from dataclasses import dataclass
from hate.constants import *
import os

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME = BUCKET_NAME
        self.ZIP_FILE_NAME = ZIP_FILE_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)
        # Notebook flow: twitter_training.csv and twitter_validation.csv
        self.TRAINING_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TRAINING_FILE)
        self.VALIDATION_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_VALIDATION_FILE)



@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRANSFORMED_FILE_NAME)
        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE 
        self.DROP_COLUMNS = DROP_COLUMNS
        self.CLASS = CLASS
        self.LABEL = LABEL
        self.TWEET = TWEET
        self.TEXT_COLUMN = TEXT_COLUMN
        self.LABELS_COLUMN = LABELS_COLUMN
        self.DROP_INGESTION_COLUMNS = DROP_INGESTION_COLUMNS
        self.COLUMN_HEADERS = COLUMN_HEADERS



@dataclass
class ModelTrainerConfig: 
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR) 
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR,TRAINED_MODEL_NAME)
        self.X_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, X_TEST_FILE_NAME)
        self.Y_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, Y_TEST_FILE_NAME)
        self.X_TRAIN_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, X_TRAIN_FILE_NAME)
        self.MAX_WORDS = MAX_WORDS
        self.MAX_LEN = MAX_LEN
        self.LOSS = LOSS
        self.METRICS = METRICS
        self.ACTIVATION = ACTIVATION
        self.LABEL = LABEL
        self.TWEET = TWEET
        self.RANDOM_STATE = RANDOM_STATE
        self.TEST_SIZE = TEST_SIZE
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        self.NUM_CLASSES = NUM_CLASSES
        self.TOKENIZER_FILE_NAME = TOKENIZER_FILE_NAME
        self.LABEL_ENCODER_FILE_NAME = LABEL_ENCODER_FILE_NAME
        self.TEXT_COLUMN = TEXT_COLUMN
        self.LABELS_COLUMN = LABELS_COLUMN
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.LSTM_UNITS = LSTM_UNITS
        self.DENSE_UNITS = DENSE_UNITS
        self.LEARNING_RATE = LEARNING_RATE
        self.DROPOUT_RATE = DROPOUT_RATE



@dataclass
class ModelEvaluationConfig: 
    def __init__(self):
        self.MODEL_EVALUATION_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR_PATH: str = os.path.join(self.MODEL_EVALUATION_MODEL_DIR,BEST_MODEL_DIR)
        self.BUCKET_NAME = BUCKET_NAME 
        self.MODEL_NAME = MODEL_NAME 



@dataclass
class ModelPusherConfig:

    def __init__(self):
        self.TRAINED_MODEL_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.BUCKET_NAME = BUCKET_NAME
        self.MODEL_NAME = MODEL_NAME
        self.TOKENIZER_FILE_NAME = TOKENIZER_FILE_NAME
        self.LABEL_ENCODER_FILE_NAME = LABEL_ENCODER_FILE_NAME
    




