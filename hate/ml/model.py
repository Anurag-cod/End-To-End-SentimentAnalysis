"""
Model architecture aligned with twitter-sentiment-analysis-with-lstm notebook:
Input -> Embedding(v+1, D) -> Dropout(0.5) -> Bidirectional(LSTM(150)) -> Dense(32, relu) -> Dense(4, softmax)
Adam(learning_rate=0.0001), categorical_crossentropy
"""
from hate.entity.config_entity import ModelTrainerConfig
from hate.constants import *
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    Dropout,
    Bidirectional,
    LSTM,
    Dense,
)
from tensorflow.keras.optimizers import Adam


class ModelArchitecture:
    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config

    def get_model(self, vocab_size: int):
        """
        Build notebook-aligned Bidirectional LSTM for 4-class sentiment.
        vocab_size should be len(tokenizer.word_index) + 1.
        """
        maxlen = self.config.MAX_LEN if self.config else MAX_LEN
        D = self.config.EMBEDDING_DIM if self.config else EMBEDDING_DIM
        lstm_units = self.config.LSTM_UNITS if self.config else LSTM_UNITS
        dense_units = self.config.DENSE_UNITS if self.config else DENSE_UNITS
        num_classes = self.config.NUM_CLASSES if self.config else NUM_CLASSES
        dropout_rate = self.config.DROPOUT_RATE if self.config else DROPOUT_RATE
        learning_rate = self.config.LEARNING_RATE if self.config else LEARNING_RATE

        inputt = Input(shape=(maxlen,))
        x = Embedding(vocab_size, D)(inputt)
        x = Dropout(dropout_rate)(x)
        x = Bidirectional(LSTM(lstm_units))(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputt, x)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)
        model.summary()
        return model
