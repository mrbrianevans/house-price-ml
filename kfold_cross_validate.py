import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy import stats
import seaborn as sns

from linear_regression import get_linear_model, train_linear_model
from neuralnet import get_mlp, train_mlp_model


def kfold_cross_validation(model, X, y):
    k = 10
    kf = KFold(n_splits=k, random_state=None)
    result = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        result.append(train_linear_model(X_train, X_test, y_train, y_test))
    result = np.array(result)
    # result = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    print("Avg neg_mean_absolute_error: {}".format(result.mean()))
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.boxplot(data=pd.DataFrame({'error': result}), y='error', ax=ax, color='#398641')
    ax.set_title('10-fold cross validation with Linear Regression')
    ax.set_ylabel('Negative Mean Absolute Error')
    plt.show()

data = pd.read_csv('detailed_house_sales.csv')
x = data[
    ['new_build_cat', 'duration_cat', 'ppdCategory_cat', 'Latitude', 'Longitude', 'Population',
     'Households',
     'Altitude', 'London zone', 'Index of Multiple Deprivation', 'Quality',
     'Distance to station', 'Average Income', 'property_type_D',
     'property_type_F', 'property_type_O', 'property_type_S',
     'property_type_T']]
y = data['price']
kfold_cross_validation(get_mlp(), x, y)
print(sorted(sklearn.metrics.SCORERS.keys()))
