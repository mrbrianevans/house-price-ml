import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

from get_data import get_scaled_Xy
from ridge_regression import train_ridge_model

seed = 31415926

def kfold_cross_validation(model, X, y, alpha=25000):
    k = 10
    kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    result = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        result.append(train_ridge_model(X_train, X_test, y_train, y_test, alpha=alpha))
    result = np.array(result)
    # result = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    # print("Avg neg_mean_absolute_error: {}".format(result.mean()))
    # fig, ax = plt.subplots(figsize=(8, 8))
    # sns.boxplot(data=pd.DataFrame({'error': result}), y='error', ax=ax, color='#398641')
    # ax.set_title('10-fold cross validation with Linear Regression')
    # ax.set_ylabel('Negative Mean Absolute Error')
    # plt.show()
    return result.mean()


x, y = get_scaled_Xy()

a_x = []
a_y = []
prev = 10000000
for alpha in range(165000, 175000, 500):
    err = kfold_cross_validation(None, x, y, alpha)
    if err > prev:
        print('------------------- ^^ BEST SCORE ^^ ---------------------------')
    prev = err
    print('For alpha=', alpha, 'error=', err)
    a_x.append(alpha)
    a_y.append(err)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(a_x, a_y)
ax.set_title('Ridge regression')
ax.set_ylabel('Mean Absolute Error')
ax.set_xlabel('alpha')
ax.ticklabel_format(useOffset=False, style='plain')
plt.savefig('images/ridge_regression_alpha_fine.png')
plt.show()
