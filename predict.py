from model import PricePredictor
from utils import csv_to_set, set_to_x_y

# Load the csv data
predicting_set, MAX_VALUE, MIN_VALUE = csv_to_set("summary.csv", recent_num=30, use_max=452.3, use_min=384.0) # Always use the same max and min as for training

# Convert the data into a format we want
x, y = set_to_x_y(predicting_set, shuffle=False, output_num=-1)

# Load the predictor
BNBUSDT_model = PricePredictor("BNB-USDT-predictor.h5")

# Make a prediction
_, predictions = BNBUSDT_model.predict(x, output_length=4) # Output length is 4 as it also predicts both the open and close prices

# Print the predictions for the next two tickers
print("============")
print("NEXT OPEN: {:.1f}".format(predictions[0] * MAX_VALUE + MIN_VALUE))
print("NEXT CLOSE: {:.1f}".format(predictions[1] * MAX_VALUE + MIN_VALUE))
print("============")
print("NEXT OPEN: {:.1f}".format(predictions[2] * MAX_VALUE + MIN_VALUE))
print("NEXT CLOSE: {:.1f}".format(predictions[3] * MAX_VALUE + MIN_VALUE))
print("============")