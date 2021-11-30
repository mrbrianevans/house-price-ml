from matplotlib import pyplot as plt

import get_data
import neuralnet
import ridge_regression

x_train, x_test, y_train, y_test = get_data.get_fixed_test_train_split()

ridge = ridge_regression.get_ridge_model(171_000)
ridge.fit(x_train, y_train)
ridge_predictions = ridge.predict(x_test)

mlp = neuralnet.get_mlp(layers=4)
mlp.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
mlp_predictions = mlp.predict(x_test)

# graphing results
qty = 500
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))
ax1.scatter(y_test[:qty], ridge_predictions[:qty], marker='.', label='Ridge regression predictions')
ax2.scatter(y_test[:qty], mlp_predictions[:qty], marker='.', label='4 layer perceptron predictions')

ax1.plot([0, 2000000], [0, 2000000], 'k--')
ax2.plot([0, 2000000], [0, 2000000], 'k--')
ax1.set_xlabel('True Target')
ax2.set_xlabel('True Target')
ax1.set_ylabel('Predicted Target')
ax1.set_title('Ridge regression')
ax2.set_title('Multilayer perceptron')
# fig.set_title(f'Comparing performance of two models')
ax1.set_xlim(0, 2000000)
ax1.set_ylim(0, 2000000)
ax2.set_xlim(0, 2000000)
ax1.ticklabel_format(useOffset=False, style='plain')
ax2.ticklabel_format(useOffset=False, style='plain')
ax2.grid()
ax2.legend()
ax1.grid()
ax1.legend()
plt.savefig(f'images/model_accuracy_comparison.png')
plt.show()
