from sklearn import linear_model

from evaluate_model import evaluate_model


def get_ridge_model(alpha=171000):
    return linear_model.Ridge(alpha=alpha)


def train_ridge_model(x_train, x_test, y_train, y_test, alpha=171000, graph=False):
    model = get_ridge_model(alpha)
    model.fit(x_train, y_train)
    e = evaluate_model(model, x_test, y_test, graph=graph, model_name='Ridge Regression')
    return e
