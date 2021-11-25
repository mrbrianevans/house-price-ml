import pandas as pd
from sklearn import linear_model

from evaluate_model import evaluate_model

x_train, y_train = pd.read_csv('data/x_train.csv').to_numpy(), pd.read_csv('data/y_train.csv')[
    'price'],

x_test, y_test = pd.read_csv(
    'data/x_test.csv').to_numpy(), pd.read_csv('data/y_test.csv')['price']
Model = linear_model.LinearRegression()
Model.fit(x_train, y_train)

evaluate_model(Model, x_test, y_test)
