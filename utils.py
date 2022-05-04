import numpy as np
import pandas as pd

def csv_to_set(csv_file, recent_num=20000, columns=[1, 4], use_max=None, use_min=None):
    print("Loading dataset...")
    dataset = pd.read_csv("summary.csv")

    print(dataset.head())

    # open_time,open_price,highest_price,lowest_price,close_price,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore
    # select the 1st and 4th columns
    training_set_raw = dataset.iloc[-recent_num:, columns].values

    if use_max is None:
        MAX_VALUE = np.max(training_set_raw)
        print("Using a max value, please use the same max value to normalise when making predictions.\nMAX_VALUE: ", MAX_VALUE)
    else:
        MAX_VALUE = use_max
    
    if use_min is None:
        MIN_VALUE = np.min(training_set_raw)
        print("Using a min value, please use the same min value to normalise when making predictions.\nMIN_VALUE: ", MIN_VALUE)
    else:
        MIN_VALUE = use_min

    # normalize the data between 0 and 1
    return (training_set_raw - MIN_VALUE) / MAX_VALUE, MAX_VALUE, MIN_VALUE

def set_to_x_y(dataset, shuffle=False, input_num=30, output_num=1):
    """
    Converts the dataset into a format we can use for training and prediction.
    Combines the open and close prices into a single vector.
    """

    x = []
    y = []

    for i in range(input_num, len(dataset) - output_num):
        new_x = []
        for j in range(input_num):
            new_x.append(dataset[i - input_num + j, 0])
            new_x.append(dataset[i - input_num + j, 1])
        x.append(new_x)
        if output_num > 0:
            new_y = []
            for j in range(output_num):
                new_y.append(dataset[i + j, 0])
                new_y.append(dataset[i + j, 1])
            y.append(new_y)

    x, y = np.array(x), np.array(y)

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    if output_num > 0:
        y = np.reshape(y, (y.shape[0], y.shape[1], 1))

    if shuffle:
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        x = x[indices]
        y = y[indices]

    return x, y