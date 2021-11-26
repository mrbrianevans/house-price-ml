import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import seaborn as sns

from linear_regression import train_linear_model
from neuralnet import train_mlp_model

splits = np.arange(0.001, 0.5, 0.1)

data = pd.read_csv('detailed_house_sales.csv')
x = data[
    ['new_build_cat', 'duration_cat', 'ppdCategory_cat', 'Latitude', 'Longitude', 'Population',
     'Households',
     'Altitude', 'London zone', 'Index of Multiple Deprivation', 'Quality',
     'Distance to station', 'Average Income', 'property_type_D',
     'property_type_F', 'property_type_O', 'property_type_S',
     'property_type_T']]
y = data['price']
iterations = 10
results = np.zeros(shape=(len(splits) * iterations, 4))
start_time = time.time()
for index in range(len(splits)):
    split = splits[index]
    for i in range(iterations):
        start = time.perf_counter_ns()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split)
        error = train_mlp_model(x_train, x_test, y_train, y_test)
        time_taken = time.perf_counter_ns() - start
        results[index*iterations+i] = (index*iterations+i, split, error, time_taken)

print('Took', time.time() - start_time, 'to run', iterations * len(splits), 'iterations')
# print(results)
error_name = 'Mean Absolute Error'
train_name = 'Training size as a proportion of full data set'
time_name = 'Time taken (nanoseconds)'
rdf = pd.DataFrame(results, columns=['i', train_name, error_name, time_name])

fig, ax = plt.subplots(figsize=(8, 8))
# sns.catplot(data=rdf, x=train_name, y=error_name, kind='box')
sns.scatterplot(data=rdf.groupby(by=train_name).mean(), x=train_name, y=error_name, hue=time_name, ax=ax, legend=False)

ax.set_xlabel('Training size as a proportion of full data set')
ax.set_ylabel(f'Mean Absolute Error (avg of {iterations} runs)')
ax.set_title('Train Test Split Performance for Multi-layer Perceptron')
plt.show()
plt.savefig('test_train_split_dropoff.png')
