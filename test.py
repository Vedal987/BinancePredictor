import numpy as np
import matplotlib.pyplot as plt

from model import PricePredictor
from utils import csv_to_set, set_to_x_y

# Load the csv data
testing_set, MAX_VALUE, MIN_VALUE = csv_to_set("summary.csv")

# Convert the data into a format we want
x, y = set_to_x_y(testing_set, shuffle=True, output_num=2)

# Test on a random sample
x_test = x[0:1]
y_test = y[0:1]

# Load the predictor
BNBUSDT_model = PricePredictor("BNB-USDT-predictor.h5")

# Predict the next 4 points
x_test, predictions = BNBUSDT_model.predict(x_test, output_length=4)
predictions = np.reshape(predictions, (predictions.shape[0], 1))

print("ACTUAL VALUES: ")
print(y_test[0] * MAX_VALUE + MIN_VALUE)
print("PREDICTED VALUES: ")
print(predictions * MAX_VALUE + MIN_VALUE)

# Construct the actual data for comparison
actual = np.copy(x_test)
actual[:, -y_test.shape[1]:] = y_test

# Plots a graph of the actual data vs the predicted data
plt.plot(np.reshape(x_test, (x_test.shape[1], 1)) * MAX_VALUE + MIN_VALUE, color = 'red', label = 'Prediction')
plt.plot(np.reshape(actual, (actual.shape[1], 1)) * MAX_VALUE + MIN_VALUE, color = 'blue', label = 'Actual')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()