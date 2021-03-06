import pickle

import pandas as pd

from data.read_data import read_data, convert_data


data_test = pd.read_csv("data/test_record.csv").values
x_test, y_test = read_data(data_test)
x_test = convert_data(x_test, 3)

model = pickle.load(open("build_model/model_LgRegress.pkl", 'rb'))

calculate_para, accuracy_para = model.evaluate_layer(x_test, y_test)
