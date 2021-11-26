import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from evaluate_model import evaluate_model


# load data
# x_train = pd.read_csv('data/x_train.csv').to_numpy()
# y_train = pd.read_csv('data/y_train.csv')['price']
# x_test, y_test = pd.read_csv('data/x_test.csv').to_numpy(), pd.read_csv('data/y_test.csv')['price']


def get_mlp():
    # multi-layer perceptron
    model = Sequential()
    feature_qty = 18
    # The Input Layer :
    model.add(
        Dense(128, kernel_initializer='normal', input_dim=feature_qty, activation='relu'))

    # The Hidden Layers :
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network :
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model


def train_mlp_model(x_train, x_test, y_train, y_test):
    model = get_mlp()
    # model.summary()

    # checkpoints
    checkpoint_name = 'checkpoints/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1,
                                 save_best_only=True,
                                 mode='auto')
    callbacks_list = [checkpoint]

    # train model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test),
              # callbacks=callbacks_list
              )

    # perform evaluation
    return evaluate_model(model, x_test, y_test)
