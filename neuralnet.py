import datetime

from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential

import get_data
from evaluate_model import evaluate_model


def get_mlp(layers=4):
    # multi-layer perceptron
    model = Sequential()
    feature_qty = 18
    # The Input Layer :
    model.add(
        Dense(128, kernel_initializer='normal', input_dim=feature_qty, activation='relu'))

    # The Hidden Layers :
    for layer in range(layers):
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network :
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model


epochs = []
errors = []


class LossAndErrorPrintingCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # print(
        #     "The average loss for epoch {} is {:7.2f} "
        #     "and mean absolute error is {:7.2f}.".format(
        #         epoch, logs["loss"], logs["mean_absolute_error"]
        #     )
        # )
        print(epoch, logs['mean_absolute_error'])
        epochs.append(epoch)
        errors.append(logs['mean_absolute_error'])


def train_mlp_model(x_train, x_test, y_train, y_test, graph=False, layers=3, iterations=4):
    print('Training MLP with', layers, 'hidden layers for', iterations, 'epochs')
    global epochs, errors
    epochs, errors = [], []
    model = get_mlp(layers=layers)
    # model.summary()

    # checkpoints
    checkpoint_name = 'checkpoints/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1,
                                 save_best_only=True,
                                 mode='auto')
    callbacks_list = [
        # checkpoint,
        LossAndErrorPrintingCallback()
    ]

    # train model
    model.fit(x_train, y_train, epochs=iterations, batch_size=32, validation_data=(x_test, y_test),
              callbacks=callbacks_list, verbose=0
              )
    print(epochs, errors)
    print('Finished at', datetime.datetime.now())
    # perform evaluation
    return evaluate_model(model, x_test, y_test, graph=graph, model_name='Multilayer Perceptron')
    # return epochs, errors
