import os

from datetime import datetime

# Common constants
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = 'hate-speech2024'
ZIP_FILE_NAME = 'dataset.zip'
LABEL = 'label'
LABELS_COLUMN = 'labels'  # notebook column name for sentiment labels
TWEET = 'tweet'
TEXT_COLUMN = 'text'  # notebook column name for tweet text


# Data ingestion constants (twitter-sentiment notebook flow)
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_TRAINING_FILE = "twitter_training.csv"
DATA_INGESTION_VALIDATION_FILE = "twitter_validation.csv"
COLUMN_HEADERS = ['Header1', 'company', 'labels', 'text']
DROP_INGESTION_COLUMNS = ['Header1', 'company']


# Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
TRANSFORMED_FILE_NAME = "final.csv"
DATA_DIR = "data"
ID = 'id'
AXIS = 1
INPLACE = True
DROP_COLUMNS = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither']
CLASS = 'class'


# Model training constants (notebook: test_size=0.2, maxlen=100, max_vocab=20000)
MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
TRAINED_MODEL_DIR = 'trained_model'
TRAINED_MODEL_NAME = 'model.h5'
X_TEST_FILE_NAME = 'x_test.csv'
Y_TEST_FILE_NAME = 'y_test.csv'
X_TRAIN_FILE_NAME = 'x_train.csv'
TOKENIZER_FILE_NAME = 'tokenizer.pickle'
LABEL_ENCODER_FILE_NAME = 'label_encoder.pickle'

RANDOM_STATE = 42
TEST_SIZE = 0.2
EPOCH = 20
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2


# Model Architecture constants (notebook: Bidirectional LSTM, 4-class softmax)
MAX_WORDS = 20000
MAX_LEN = 100
NUM_CLASSES = 4
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
ACTIVATION = 'softmax'
EMBEDDING_DIM = 100
LSTM_UNITS = 150
DENSE_UNITS = 32
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5


# Model  Evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
BEST_MODEL_DIR = "best_Model"
MODEL_EVALUATION_FILE_NAME = 'loss.csv'


MODEL_NAME = 'model.h5'
APP_HOST = "0.0.0.0"
APP_PORT = 8080
