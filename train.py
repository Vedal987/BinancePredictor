from model import PricePredictor
from utils import csv_to_set, set_to_x_y

# Load the csv data (the second two values are the max value and min value which are uneeded in this case)
training_set, _, _ = csv_to_set("summary.csv")

# Convert the data into a format we want
x, y = set_to_x_y(training_set, shuffle=True)

# Training set is the first 4000 samples
x_train = x[:4000]
y_train = y[:4000]

# Validation set is the last 1000 samples
x_val = x[-1000:]
y_val = y[-1000:]

# Construct the predictor
BNBUSDT_model = PricePredictor()

# Train it
BNBUSDT_model.train(x_train, y_train, x_val, y_val)

# Save it
BNBUSDT_model.save("BNB-USDT-predictor.h5")