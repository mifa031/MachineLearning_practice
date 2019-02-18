import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization, Flatten
from keras.optimizers import Nadam


class PriceNetwork:
    def __init__(self, seq_length=0, data_dim=0, output_dim=0, lr=0.01):
        # LSTM 신경망
        self.model = Sequential()

        self.model.add(LSTM(256, input_shape=(seq_length, data_dim),
                            return_sequences=False, stateful=False))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(output_dim, activation='linear'))

        self.model.summary()

        self.model.compile(optimizer=Nadam(lr=lr), loss='mse')
        
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)
