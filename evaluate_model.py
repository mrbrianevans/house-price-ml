from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error


def evaluate_model(model, x_test, y_test, qty=5000, graph=False, model_name='Regression'):
    predictions = model.predict(x_test)

    error = mean_absolute_error(y_test, predictions)
    # plot the first 5,000 predictions to make them easier to compare with the true target values
    if graph:
        print(model_name, 'Mean absolute error', round(error))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test[:qty], predictions[:qty], marker='.')
        ax.set_xlabel('True Target')
        ax.set_ylabel('Predicted Target')
        ax.set_title(f'Performance of {model_name} model with unseen data')
        ax.set_xlim(0, 2000000)
        ax.set_ylim(0, 2000000)
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.grid()
        plt.savefig(f'images/model_accuracy_{model_name}.png')
        plt.show()
    return error
