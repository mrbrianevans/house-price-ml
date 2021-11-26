import pandas as pd
from sklearn import linear_model

from evaluate_model import evaluate_model


# x_train, y_train = pd.read_csv('data/x_train.csv').to_numpy(), pd.read_csv('data/y_train.csv')[
#     'price'],
#
# x_test, y_test = pd.read_csv(
#     'data/x_test.csv').to_numpy(), pd.read_csv('data/y_test.csv')['price']

def get_linear_model():
    return linear_model.LinearRegression()


def train_linear_model(x_train, x_test, y_train, y_test):
    model = get_linear_model()
    model.fit(x_train, y_train)

    return evaluate_model(model, x_test, y_test)
