import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

import get_data
import neuralnet
import ridge_regression

seed = 3420

X, y = get_data.get_scaled_Xy()

k = 10
kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
ridge_result = []
mlp_result = []
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ridge_result.append(ridge_regression.train_ridge_model(X_train, X_test, y_train, y_test))
    mlp_result.append(neuralnet.train_mlp_model(X_train, X_test, y_train, y_test))

df = pd.DataFrame(
    {'error': np.array(ridge_result + mlp_result),
     'model': ['Ridge regression' for i in ridge_result] + ['4 layer perceptron' for j in
                                                            mlp_result]})

fig, ax = plt.subplots(figsize=(8, 8))
sns.boxplot(data=df,
            y='error', x='model', ax=ax, color='#398641')
ax.set_title('10-fold cross validation')
ax.set_ylabel('Mean Absolute Error')
plt.savefig('images/cross_validation_boxplot.png')
plt.show()
