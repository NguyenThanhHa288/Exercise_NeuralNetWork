import pickle

import pandas as pd

from data.read_data import read_data, convert_data
from model.layer_model import sequential

"""
Build model Logistic Regression to Data train
"""

data = pd.read_csv("../data/train_record.csv").values

x_train, y_train = read_data(data)
x_train = convert_data(x_train, 3)

model = sequential()

model.input(x_train, y_train)
model.conv2d(number=64, activation="sigmoid")
model.conv2d(number=32, activation="sigmoid")
model.conv2d(number=16, activation="sigmoid")
model.fit(learning_rate=0.02, epochs=20000, verbose=1000)


# pickle.dump(model, open('model_LgRegress.pkl', 'wb'))
