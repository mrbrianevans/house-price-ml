from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mdf = pd.read_csv('detailed_house_sales.csv')
pdf = pd.get_dummies(mdf, columns=['property_type'])

x = pdf[['new_build_cat', 'duration_cat', 'ppdCategory_cat', 'Latitude', 'Longitude', 'Population',
         'Households',
         'Altitude', 'London zone', 'Index of Multiple Deprivation', 'Quality',
         'Distance to station', 'Average Income', 'property_type_D',
         'property_type_F', 'property_type_O', 'property_type_S',
         'property_type_T']]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(x.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=x.columns)
x = df_scaled
y = pdf['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal', input_dim=x.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

# checkpoints
checkpoint_name = 'checkpoints/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True,
                             mode='auto')
callbacks_list = [checkpoint]

NN_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2,
             callbacks=callbacks_list)

predictions = NN_model.predict(x_test)

print('Mean absolute error', round(mean_absolute_error(y_test, predictions)))
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(y_test, predictions, marker='.')
ax.set_xlabel('True Target')
ax.set_ylabel('Predicted Target')
ax.set_xlim(0, 2000000)
ax.set_ylim(0, 2000000)
plt.savefig('keras_results.png')
