from matplotlib import pyplot as plt

import get_data
import neuralnet

x, y = get_data.get_scaled_Xy()

x_train, x_test, y_train, y_test = get_data.get_fixed_test_train_split()

# ridge_regression.train_ridge_model(x_train, x_test, y_train, y_test, graph=True, alpha=171000)

# epochs2, errors2 = neuralnet.train_mlp_model(x_train, x_test, y_train, y_test, graph=True, layers=2,
#                                              iterations=10)
# epochs3, errors3 = neuralnet.train_mlp_model(x_train, x_test, y_train, y_test, graph=True, layers=3,
#                                              iterations=10)
# epochs4, errors4 = neuralnet.train_mlp_model(x_train, x_test, y_train, y_test, graph=True, layers=4,
#                                              iterations=10)
# epochs5, errors5 = neuralnet.train_mlp_model(x_train, x_test, y_train, y_test, graph=True, layers=5,
#                                              iterations=10)

x_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_points = [136647.59375, 131050.4453125, 130320.4375, 128989.8046875, 126933.5625, 124781.3515625, 123278.5, 121762.6171875, 120603.3671875,
            119705.15625, 134252.484375, 128514.9609375, 124291.7265625, 120439.5859375, 118645.984375, 117607.3984375, 116825.671875, 116156.1875,
            115596.8125, 115025.6953125, 132144.0625, 123624.875, 120037.5703125, 118465.25, 117289.2890625, 116189.59375, 115321.78125, 114671.109375,
            114268.78125, 113866.421875, 130106.9921875, 121779.6953125, 119236.7734375, 117752.984375, 116497.5625, 115632.7578125, 115001.34375, 114294.6171875, 113778.8359375, 113463.9140625]

fig, ax = plt.subplots(figsize=(8, 8))
# ax.scatter([0, 1, 2, 3, 4], [134335.84375, 128820.0859375, 123936.40625, 120247.4453125, 118487.71875])
ax.scatter(x_points, y_points[0:10], label='2 hidden layers')
ax.scatter(x_points, y_points[10:20], label='3 hidden layers')
ax.scatter(x_points, y_points[20:30], label='4 hidden layers')
ax.scatter(x_points, y_points[30:], label='5 hidden layers')
ax.set_xlabel('Epochs of training')
ax.set_ylabel('Mean absolute error')
ax.set_title(f'Multilayer perceptron performance')
plt.legend()
plt.savefig('images/mlp_epochs.png')
plt.show()


