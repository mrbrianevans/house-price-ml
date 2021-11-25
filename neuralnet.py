import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from evaluate_model import evaluate_model

# load data
x_train = pd.read_csv('data/x_train.csv').to_numpy()
y_train = pd.read_csv('data/y_train.csv')['price']
x_test, y_test = pd.read_csv('data/x_test.csv').to_numpy(), pd.read_csv('data/y_test.csv')['price']

# multi-layer perceptron
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal', input_dim=x_train.shape[1], activation='relu'))

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

# train model
NN_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2,
             callbacks=callbacks_list)

# perform evaluation
evaluate_model(NN_model, x_test, y_test)
