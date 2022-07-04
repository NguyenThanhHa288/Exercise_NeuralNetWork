import pickle

import pandas as pd

from data.read_data import read_data, convert_data
from model.layer_model import sequential

data = pd.read_csv("../data/train_record.csv").values

x_train, y_train = read_data(data)
x_train = convert_data(x_train, 4)

model = sequential()

model.input(x_train, y_train)
model.conv2d(number=64, activation="relu")
model.conv2d(number=32, activation="relu")
model.conv2d(number=16, activation="relu")
model.conv2d(number=8, activation="relu")
model.conv2d(number=4, activation="sigmoid")
model.fit(learning_rate=0.01, epochs=30, verbose=2)

# pickle.dump(model, open('model_LgRegress.pkl', 'wb'))
