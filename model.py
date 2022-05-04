import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import os

# UNCOMMENT THIS LINE TO FORCE CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class PricePredictor:
    def __init__(self, load=None, input_num=30, output_num=1):
        print("Initialising Price Predictor...")
        # If the load value is given, try to load the model from a file
        if load is not None:
            print(f"Loading model from {load}...")
            self.model = load_model(load)
        else:
            print("Creating new model...")
            self.model = Sequential()
            self.model.add(LSTM(units = 64, return_sequences = True, input_shape = (input_num * 2, 1)))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units = 64, return_sequences = True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units = 64, return_sequences = True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units = 64))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units = output_num * 2))

            self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        self.input_num = input_num
        self.output_num = output_num
        print(self.model.summary())
        print("Model finished loading.")

    def train(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=32):
        self.model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_val, y_val))

    def save(self, path):
        self.model.save(path)
        print(f"Model saved at {path}")

    def predict(self, x, output_length=4):
        predictions = []
        i = -1

        while len(predictions) < output_length:
            output = self.model.predict(x[:, -self.input_num * 2:])[0]
            # print(x_test[0, -60:])
            # print("=>")
            # print(output)
            for out in output:
                predictions.append(out)
                i += 1
                x = np.reshape(x, (x.shape[1], 1))
                x = np.concatenate([x, np.reshape(predictions[i], (1,1))])
                x = np.reshape(x, (1, x.shape[0], 1))

        predictions = np.array(predictions)
        # Returns the full input + predictions and just the predictions separately
        return x, predictions
                