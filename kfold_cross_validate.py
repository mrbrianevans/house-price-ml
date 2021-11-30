import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
import seaborn as sns

import get_data
import neuralnet
import ridge_regression

seed = 3420

def kfold_cross_validation(train_model, X, y, graph=False):
    k = 10
    kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    result = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        result.append(train_model(X_train, X_test, y_train, y_test))
    result = np.array(result)
    # result = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    if graph:
        print("Avg neg_mean_absolute_error: {}".format(result.mean()))
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.boxplot(data=pd.DataFrame({'error': result}), y='error', ax=ax, color='#398641')
        ax.set_title('10-fold cross validation with Linear Regression')
        ax.set_ylabel('Negative Mean Absolute Error')
        plt.show()
    return result.mean()

x, y = get_data.get_scaled_Xy()

kfold_cross_validation(neuralnet.train_mlp_model, x, y, True)
