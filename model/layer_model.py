from model.model import NeuralNetwork


class sequential:

    """
    Layer of model Logistic Regression
    """

    model = NeuralNetwork()

    def __init__(self):
        self.x_train = None
        self.y_train = None

    def input(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        n, d = x_train.shape
        sequential.model.get_layer(d, "")

    @staticmethod
    def conv2d(number: int, activation: str):
        sequential.model.get_layer(number, activation)

    def fit(self, learning_rate, epochs, verbose):
        sequential.model.fit(self.x_train, self.y_train, learning_rate, epochs, verbose)

    @staticmethod
    def evaluate_layer(x_test, y_test):
        calculate_para, accuracy_para = sequential.model.predict_test(x_test, y_test)
        return calculate_para, accuracy_para

    @staticmethod
    def predict_layer(x):
        result = sequential.model.predict_object(x)
        return result
