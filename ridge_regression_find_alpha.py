import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from get_data import get_scaled_Xy
from linear_regression import get_linear_model
from ridge_regression import get_ridge_model
from ridge_regression import train_ridge_model

x, y = get_scaled_Xy()
model = get_ridge_model()
r_result = cross_val_score(model, x, y, cv=10, scoring='neg_mean_absolute_error', n_jobs=10)
print('Cross validated ridge error:', r_result.mean())
model = get_linear_model()
l_result = cross_val_score(model, x, y, cv=10, scoring='neg_mean_absolute_error')

print('Cross validated linear error:', l_result.mean(), l_result)

difference = (l_result.mean() - r_result.mean()) / l_result.mean() * 100

print('Ridge regression outperformed linear regression by', difference, '% with cross validation')

seed = 31415926


def ridge_kfold_cross_validation(X, y, alpha=171000, k=10):
    kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    result = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        result.append(train_ridge_model(X_train, X_test, y_train, y_test, alpha=alpha))
    result = np.array(result)
    return result.mean()


a_x = []
a_y = []
prev = 10000000
for alpha in range(165000, 175000, 500):
    err = ridge_kfold_cross_validation(x, y, alpha)
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
