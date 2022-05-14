"""
ML-Train: A code to solve an input-output problem with the CNN+RNN(LSTM)

features:
1. dataset split module
2. extensible sequential keras module
3. Learning Rate Schedule
4. statistical analysis module and regression function
5. visualization module
6. sampling module for test dataset
7. function of "repacking and reshape VOF-Matrix"
8. time clock of program running
9. New Repacking and Reshape Algorithms

** results documents should be .PDF instead of .PNG

"""

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sklearn
from tensorflow import keras
from keras import models
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Global variable
# Sample interval in test set for prediction
NUM_TEST_SAMPLE_POINTS = 1


class EvaluateLossMaeCallback(keras.callbacks.Callback):
    @staticmethod
    def on_test_batch_end(batch, logs=None):
        print("For batch {}, loss is {:7.5e}. mae is {:7.5f}".format(batch, logs["loss"], logs["mae"]))


def CNNLSTM():
    # build a CNN architecture
    inputs = Input(shape = (3, 3, 1), name = 'input_layer')

    norm = layers.Normalization(mean = 0., variance = 1.)(inputs)

    conv1 = layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = 'relu',
        padding = "same",
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
        name = 'convolution_layer_1')(norm)

    conv2 = layers.Conv2D(
        filters = 64,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = 'relu',
        padding = "same",
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
        name = 'convolution_layer_2')(conv1)

    conv3 = layers.Conv2D(
        filters = 128,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = 'relu',
        padding = "same",
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
        name = 'convolution_layer_3')(conv2)

    pooling = layers.MaxPooling2D(
        pool_size = (1, 1),
        strides = (1, 1),
        padding = "valid",
        name = 'Max_pooling3')(conv3)

    flatten = layers.Flatten(name = 'Flatten')(pooling)

    dim_expand = layers.Lambda(
        lambda flatten: tf.expand_dims(flatten, axis = 1))(flatten)

    # LSTM layers
    LSTM = layers.LSTM(
        units = 50,
        activation = 'tanh',
        recurrent_activation = 'relu',
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
        dropout = 0.0,
        recurrent_dropout = 0.0,
    )(flatten)

    # Regression output layer
    outputs = Dense(1,
                    kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
                    name = 'output_layer')(LSTM)

    CNNLSTM = Model(inputs = inputs,
                    outputs = outputs,
                    name = 'CNNLSTM')

    return CNNLSTM


def scheduler(epoch):
    """
    config learning rate scheduler
    Args:
        epoch: training epoch

    Returns: changeable learning rate

    """
    if epoch < 1000:
        return 1e-3

    elif epoch < 400:
        return 9e-5

    elif epoch < 600:
        return 7e-5

    elif epoch < 800:
        return 5e-5

    else:
        return 3e-5


def regress_equ(b1, b2):
    """
    display func for linear regression y= b1 * x + b
    Args:
        b1: slope of the linear regression line
        b2: intercept of the linear regression line

    Returns: a callable format for texting

    """
    if b2 < 0:
        return 'Prediction = {0} * Target - {1}'.format(str('%.3f' % b1), str('%.3e' % abs(b2)))
    else:
        return 'Prediction = {0} * Target + {1}'.format(str('%.3f' % b1), str('%.3e' % b2))


if __name__ == "__main__":

    # Start the clock
    start = time.perf_counter()

    # Load the generated data and obtain the values for x, t:
    data = scipy.io.loadmat('Data/database.mat')  # Import dataset
    x = data["datafra"]  # 188232 x 9 double： fraction of volume
    t = data["datacur"]  # 188232 x 1 double:  generated curvature

    # Repacking and reshape VOF-Matrix x as a 3x3 Matrix:
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = tf.split(x, num_or_size_splits = 9, axis = 1)
    x_h1 = tf.concat([x7, x8, x9], axis = 1)
    x_h1 = tf.expand_dims(x_h1, axis = -1)
    x_h2 = tf.concat([x4, x5, x6], axis = 1)
    x_h2 = tf.expand_dims(x_h2, axis = -1)
    x_h3 = tf.concat([x1, x2, x3], axis = 1)
    x_h3 = tf.expand_dims(x_h3, axis = -1)
    x_reshape = tf.stack([x_h1, x_h2, x_h3], axis = 1)
    x = np.reshape(np.array(x_reshape), (188232, 3, 3, 1))

    # Process the raw data
    # Split the dataset to 70% Training set, 15% test set and 15% validation set
    X_train, X_test, t_train, t_test = train_test_split(x, t, test_size = 0.3, random_state = 321)
    X_test, X_val, t_test, t_val = train_test_split(X_test, t_test, test_size = 0.5, random_state = 321)

    disp_Datashape = [["Input data", str(x.shape)], ["Output data", str(t.shape)],
                      ["X_train", str(X_train.shape)], ["t_train", str(t_train.shape)],
                      ["X_test", str(X_test.shape)], ["t_test", str(t_test.shape)],
                      ["X_val", str(X_val.shape)], ["t_val", str(t_val.shape)]]

    # Print the name of dataset and datashape
    print("=================================================================")
    print("\t{:<25} {:<25}".format('Name', 'Datashape'))
    print("-----------------------------------------------------------------")
    for i in disp_Datashape:
        name, datashape = i
        if str(name[0]) == 'X':
            print("-----------------------------------------------------------------")
        print("\t{:<25} {:<25}".format(name, datashape))
    print("=================================================================")

    # Choose framework for CNN
    model = CNNLSTM()

    # Print the NN architecture and the NN topology
    model.summary()
    keras.utils.plot_model(model, to_file = 'CNN_architecture.png', show_shapes = True)

    # Configurate the training parameters
    model.compile(optimizer = keras.optimizers.Adam(beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8),
                  loss = tf.losses.MeanSquaredError(),
                  metrics = ['mae'],
                  run_eagerly = False)

    # Start of training process
    print("Fit model on training data")

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(X_train,
                        t_train,
                        epochs = 1000,
                        verbose = 2,
                        batch_size = 64,  # 64 best
                        validation_data = (X_val, t_val),
                        validation_batch_size = 32,
                        callbacks = [callback]
                        )

    # Evaluate the model with test dataset
    print("\nEvaluate on test data")

    results = model.evaluate(X_test,
                             t_test,
                             batch_size = 32,
                             verbose = 0,
                             callbacks = None
                             )

    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(mae) + 1)
    print("test loss: {}, test mae: {}".format(results[0], results[1]))

    # Prediction in test dataset
    print("\nGenerate predictions")

    # interval sampling from test dataset
    X_pred = X_test[::NUM_TEST_SAMPLE_POINTS]
    t_pred_true = t_test[::NUM_TEST_SAMPLE_POINTS]
    t_pred = model.predict(X_pred)

    # The mean error of regression
    mean_error = abs(np.mean(t_pred - t_pred_true))

    print("number of sample points: {}\n".format(str(len(X_pred))),
          "prediction shape: {} target shape: {}".format(str(t_pred.shape), str(t_pred_true.shape)))
    print("\nmean error of regression:", mean_error)

    # End the clock
    end = time.perf_counter()
    print("\n runtime: %s second " % (end - start))

    # Statistic analyse: calculate relationship coefficient R and determination coefficient R^2
    R_2 = sklearn.metrics.r2_score(t_pred_true, t_pred)
    R = R_2 ** 0.5

    # Statistic analyse: linear regression with "Least Squares"
    t_pred_true_reshape = np.reshape(t_pred_true[:len(t_pred_true)], (len(t_pred_true),))
    t_pred_reshape = np.reshape(t_pred[:len(t_pred)], (len(t_pred),))

    Target = np.vstack([t_pred_true_reshape,
                        np.ones(len(t_pred_true_reshape))
                        ]).T
    Output = t_pred_reshape
    beta1, beta2 = np.linalg.lstsq(Target, Output, rcond = None)[0]

    # Plotting the result
    # plotting the convergence history
    plt.figure('convergence history')
    plt.title('convergence history')
    plt.xlim(0, 1000)
    plt.xticks(np.arange(0, 1100, 200))
    plt.xlabel('Epochs')
    plt.ylabel('log(loss)')

    plt.semilogy(epochs, history.history['val_loss'], c = 'orange', label = 'validation loss')
    plt.semilogy(epochs, history.history['loss'], c = 'blue', label = 'training loss', linewidth = 2)
    plt.legend()

    # Save figure：the convergence history
    plt.savefig('convergence_history_cnn.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()

    # Plotting the model accuracy
    plt.figure('training accuracy')
    plt.title('training accuracy')
    plt.plot(epochs, val_mae, color = 'orange', label = 'Validation mae')
    plt.plot(epochs, mae, color = 'b', label = 'Training mae')
    plt.xlim(0, 1000)
    plt.xticks(np.arange(0, 1100, 200))
    plt.xlabel('training epochs')
    plt.ylabel('MeanAbsoluteError')
    plt.legend()

    # Save figure：the model accuracy
    plt.savefig('training_accuracy_cnn.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()

    # Plotting the prediction VS target
    plt.figure('Prediction vs Target')
    plt.title('Prediction vs Target')
    plt.xlabel('Target')
    plt.ylabel('Prediction')

    # Set x and y axes to the same scale
    plt.xlim(-0.25, 0.25)
    plt.ylim(-0.25, 0.25)
    plt.scatter(t_pred_true, t_pred, marker = 'o', c = 'blue', label = 'Data')

    # reference line for Y=T and regression line with "Least Squares"
    x_ref = np.linspace(-2.5, 2.5, 10)
    y_ref = x_ref
    plt.plot(x_ref, y_ref, color = 'black', linestyle = '--', label = 'Y=T')
    plt.plot(x_ref, beta1 * x_ref + beta2, color = 'red', label = 'Fit')

    # display R, R^2 and linear regression function
    plt.text(-0.24, 0.12, regress_equ(beta1, beta2))
    plt.text(-0.24, 0.09, r'$R^2=$')
    plt.text(-0.205, 0.09, '%.5f' % R_2)
    plt.text(-0.24, 0.06, r'$R \ \, =$')
    plt.text(-0.205, 0.06, '%.5f' % R)
    plt.legend()

    # Save figure：the prediction VS target
    plt.savefig('Prediction_vs_Target_cnn.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()
