import numpy as np
from sklearn.model_selection import KFold

from support_model.activation_model import sigmoid, sigmoid_derivative, relu_derivative, relu
from support_model.evaluate_model import calculate_loss, accuracy


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.activation = []
        self.optimal_val = 0

        self.optimal_w = []
        self.optimal_b = []
        self.w = []
        self.b = []

    def get_layer(self, number: int, activation: str):
        self.layers.append(number)
        if activation == "sigmoid":
            self.activation.append(0)
        elif activation == "tanh":
            self.activation.append(1)
        elif activation == "relu":
            self.activation.append(2)

    def layer_model(self):
        self.layers.append(1)
        for i in range(0, len(self.layers) - 1):
            w_unit = np.random.rand(self.layers[i], self.layers[i + 1])
            b_unit = np.zeros((self.layers[i + 1], 1))
            self.w.append(w_unit / self.layers[i])
            self.b.append(b_unit)

    def fit_partial(self, x, y, learning_rate):
        z = [x]
        out = z[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.w[i]) + self.b[i].T)
            z.append(out)

        y = y.reshape(-1, 1)
        dz = [-(y / z[-1] - (1 - y) / (1 - z[-1]))]
        dw = []
        db = []

        for count in reversed(range(0, len(self.layers) - 1)):
            if count == (len(self.layers) - 2):
                dw_ = np.dot((z[count]).T, dz[-1] * sigmoid_derivative(z[count + 1]))
                db_ = (np.sum(dz[-1] * sigmoid_derivative(z[count + 1]), 0)).reshape(-1, 1)
                dz_ = np.dot(dz[-1] * sigmoid_derivative(z[count + 1]), self.w[count].T)
            else:
                dw_ = np.dot((z[count]).T, dz[-1] * relu_derivative(z[count + 1]))
                db_ = (np.sum(dz[-1] * relu_derivative(z[count + 1]), 0)).reshape(-1, 1)
                dz_ = np.dot(dz[-1] * relu_derivative(z[count + 1]), self.w[count].T)
            dw.append(dw_)
            db.append(db_)
            dz.append(dz_)

        dw = dw[::-1]
        db = db[::-1]

        for i in range(0, len(self.layers) - 1):
            self.w[i] = self.w[i] - learning_rate * (dw[i] / len(y))
            self.b[i] = self.b[i] - learning_rate * (db[i] / len(y))

    def fit(self, x, y, learning_rate, epochs=30, verbose=2):
        self.layer_model()

        global acr_train, acr_val

        k_fold = KFold(n_splits=8, shuffle=True)

        for train_ids, val_ids in k_fold.split(x, y):

            for epoch in range(0, epochs):
                self.fit_partial(x[train_ids], y[train_ids], learning_rate)

                if epoch % verbose == 0:
                    loss, acr_train = self.predict(x[train_ids], y[train_ids])
                    loss_val, acr_val = self.predict(x[val_ids], y[val_ids])

                    print("Epoch {}, loss {}, accuracy {}".format(epoch, loss, acr_train))
                    print("Epoch {}, loss_val {}, accuracy_val {}".format(epoch, loss_val, acr_val))

            if acr_val > self.optimal_val:
                self.optimal_w = self.w
                self.optimal_b = self.b

    def predict(self, x, y):
        for count in range(0, len(self.layers) - 1):
            if count == (len(self.layers) - 2):
                x = sigmoid(np.dot(x, self.w[count]) + self.b[count].T)
            else:
                x = relu(np.dot(x, self.w[count]) + self.b[count].T)
        calculate_para = calculate_loss(y, x)
        accuracy_para = accuracy(y, x)
        return calculate_para, accuracy_para

    def predict_test(self, x, y):
        for count in range(0, len(self.layers) - 1):
            if count == (len(self.layers) - 2):
                x = sigmoid(np.dot(x, self.optimal_w[count]) + self.optimal_b[count].T)
            else:
                x = relu(np.dot(x, self.optimal_w[count]) + self.optimal_b[count].T)
        calculate_para = calculate_loss(y, x)
        accuracy_para = accuracy(y, x)
        return calculate_para, accuracy_para

    def predict_object(self, x):
        for count in range(0, len(self.layers) - 1):
            if count == (len(self.layers) - 2):
                x = sigmoid(np.dot(x, self.optimal_w[count]) + self.optimal_b[count].T)
            else:
                x = relu(np.dot(x, self.optimal_w[count]) + self.optimal_b[count].T)
        return x