from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error


def evaluate_model(model, x_test, y_test, qty=5000, graph=False):
    predictions = model.predict(x_test)

    error = mean_absolute_error(y_test, predictions)
    # print('Mean absolute error', round(error))
    # plot the first 5,000 predictions to make them easier to compare with the true target values
    if graph:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(y_test[:qty], predictions[:qty], marker='.')
        ax.set_xlabel('True Target')
        ax.set_ylabel('Predicted Target')
        ax.set_xlim(0, 2000000)
        ax.set_ylim(0, 2000000)
        plt.savefig('model_accuracy.png')
    return error
