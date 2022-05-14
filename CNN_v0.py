"""
ML-Train
A code to solve an input-output problem with the Convolutional Neural Network
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas
import sklearn
from tensorflow import keras
from keras import models
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from IPython.display import Image


class Encoder(layers.Layer):

    def __init__(self, l2_rate=1e-2):
        super().__init__()
        self.l2_rate = l2_rate

    def build(self, input_shape):
        self.Dense1 = layers.Dense(
            units=32,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )
        self.Dense2 = layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )
        self.Dense3 = layers.Dense(
            units=128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )

    def call(self, inputs, **kwargs):
        dense1_layer = self.Dense1(inputs)
        dense2_layer = self.Dense2(dense1_layer)
        dense3_layer = self.Dense3(dense2_layer)
        return dense3_layer


class Decoder(layers.Layer):
    def __init__(self, l2_rate=1e-3):
        super().__init__()
        self.l2_rate = l2_rate

    def build(self, input_shape):
        self.Dense1 = layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )
        self.Dense2 = layers.Dense(
            units=32,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )
        self.Dense3 = layers.Dense(
            units=16,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )

    def call(self, inputs, **kwargs):
        dense1_layer = self.Dense1(inputs)
        dense2_layer = self.Dense2(dense1_layer)
        dense3_layer = self.Dense3(dense2_layer)
        return dense3_layer


# unfinished class
class Physics_Embedded_Decoder(layers.Layer):
    def __init__(self, l2_rate=1e-3):
        super().__init__()
        self.l2_rate = l2_rate

    def build(self, input_shape):
        self.Dense1 = layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )

        self.BCs = layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )

        self.Derivatives = layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )

        self.CURL =layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )

    def call(self, inputs, **kwargs):
        dense1_layer = self.Dense1(inputs)
        BCs = self.BCs(dense1_layer)
        Derivatives = self.Derivatives(BCs)
        Define_Operator = self.CURL(Derivatives)
        Outputs = Define_Operator
        return Outputs


class CAE(keras.Model):
    def __init__(self):
        super().__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()

    def call(self, inputs):
        input_encoder = self.Encoder(inputs)
        decoder_output = self.Decoder(input_encoder)
        return decoder_output


class PhyCAE(keras.Model):
    def __init__(self):
        super().__init__()
        self.Encoder = Encoder()
        self.PhyDecoder = Physics_Embedded_Decoder()

    def call(self, inputs):
        input_encoder = self.Encoder(inputs)
        phy_decoder_output = self.PhyDecoder(input_encoder)
        return phy_decoder_output


if __name__ == "__main__":

    model = CAE()
    model.build((None, 16))
    model.summary()
    print(model.layers)
    print(model.weights)


    # load the generated data and obtain the values for x, t:
    data = scipy.io.loadmat('Master Thesis/Data/database.mat')  # Import dataset
    x = data["datafra"]  # 188232 x 9 double： fraction of volume
    t = data["datacur"]  # 188232 x 1 double:  generated curvature

    # Repacking and reshape VOF-Matrix x as a 3x3 Matrix:
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = tf.split(x, num_or_size_splits=9, axis=1)
    x_h1 = tf.concat([x7, x8, x9], axis=1)
    x_h2 = tf.concat([x4, x5, x6], axis=1)
    x_h3 = tf.concat([x1, x2, x3], axis=1)
    x_reshape = tf.stack([x_h1, x_h2, x_h3])  # 3x188232x3
    x_ndarray = np.reshape(np.array(x_reshape), (188232, 9))

    '''
     # Standardization for x and t
     scaler = StandardScaler()
     x_scaled = scaler.fit_transform(x)
     t_scaled = scaler.fit_transform(t)
     '''

    # Process the raw data
    # split the dataset to 70% Training set, 15% test set and 15% validation set
    X_train, X_test, t_train, t_test = train_test_split(x_reshape, t, test_size=0.2, random_state=321)
    X_test, X_val, t_test, t_val = train_test_split(X_test, t_test, test_size=0.5, random_state=321)
    disp_Datashape = [["Input data", str(x_reshape.shape)], ["Output data", str(t.shape)],
                      ["X_Train", str(X_train.shape)], ["t_Train", str(t_train.shape)],
                      ["X_Test", str(X_test.shape)], ["t_test", str(t_test.shape)],
                      ["X_Val", str(X_val.shape)], ["t_val", str(t_val.shape)]]
    print("{:<15} {:<15}".format('Name', 'Datashape'))
    for i in disp_Datashape:
        name, datashape = i
        print("{:<15} {:<15}".format(name, datashape))

    # build the training model with sequential model
    inputs = Input(shape=(3, 188232, 3), name='input_layer')

    CNN = Model(inputs=inputs,
                outputs=outputs,
                name='feed_forward_NN')

    # print the NN architecture and the NN topology
    CNN.summary()
    keras.utils.plot_model(CNN, to_file='architectureCNN.png')
    Image('architectureCNN.png')

    # Configurate the training method
    CNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
                loss=tf.losses.MeanSquaredError(),
                metrics=['mae'],
                run_eagerly=False)

    print("Fit model on training data")
    history = CNN.fit(X_train,
                      t_train,
                      epochs=100,
                      verbose=2,
                      batch_size=32,
                      validation_data=(X_val, t_val),
                      )

    # Evaluate the model with test dataset
    print("Evaluate on test data")
    results = CNN.evaluate(X_val, t_val, batch_size=16)
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(mae) + 1)
    print("test loss, test mae:", results)

    # Prediction with 20 sample points in test dataset
    print("Generate predictions")
    predictions = CNN.predict(X_test)
    print("prediction shape:", predictions.shape)

    # Plotting the result
    # plotting the convergence history
    plt.figure('convergence history')
    plt.title('convergence history')
    plt.xlabel('Iterations')
    plt.ylabel('log(loss)')
    plt.legend('loss')
    plt.semilogy(epochs, history.history['loss'], c='blue', label='training loss')
    plt.semilogy(epochs, history.history['val_loss'], c='orange', label='validation loss')
    plt.legend()
    plt.show()

    # Plotting the model accuracy
    plt.figure('training accuracy')
    plt.title('training accuracy')
    plt.plot(epochs, mae, color='b', label='Training mae')
    plt.plot(epochs, val_mae, color='orange', label='Validation mae')
    plt.xlabel('training epochs')
    plt.ylabel('MeanSquaredError')
    plt.legend()
    plt.show()

    # Plotting the prediction
    plt.figure('prediction output')
    plt.title('prediction output')
    plt.xlabel('Input Testdata')
    plt.ylabel('predicted curvature')
    plt.legend('predicted data')
    plt.scatter(np.arange(100), t_test[:100], marker='o', c='orange', label='target value from test set')
    plt.scatter(np.arange(100), predictions[:100], marker='o', c='blue', label='predict output from NN')
    plt.legend()
    plt.show()

    """
    activation function
    xa = np.linspace(-10, 10, 2000)
    ya = xa
    ax = plt.gca()
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer = True))
    plt.plot(xa, ya, label = 'tanh', c='r')
    plt.savefig('relu.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()
    """

