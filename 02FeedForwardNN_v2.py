"""
ML-Train: A code to solve an input-output problem with the Feed forward Neural Network

features:
1. dataset split module
2. extensible sequential keras module
3. Learning Rate Schedule
4. statistical analysis module and regression function
5. visualization module
6. sampling module for test dataset
7. function of "repacking and reshape VOF-Matrix"
8. time clock of ML-train
9. Error histogram for trained model in different dataset
10. Predict vs Target for all data

** results documents should be .PDF instead of .PNG

Author： Junjie Zhang  Version: 2.0
"""

import time
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.io
import sklearn
from tensorflow import keras
from keras import models
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

# Sample interval in test set for prediction
NUM_TEST_SAMPLE_POINTS = 1


def scheduler(epoch):
    """
    config learning rate scheduler
    Args:
        epoch: training epoch

    Returns: changeable learning rate

    """
    if epoch < 1000:
        return 9.01e-4  # 4000batch 1e-4 overfitting --> 7.5e-5 underfitting --> 1e-5, 5e-5 learn nothing, 8000batch 1e-4, 3e-4 low acc
    elif epoch < 400:
        return 1e-4
    elif epoch < 600:
        return 9e-5
    elif epoch < 800:
        return 7e-5
    else:
        return 5e-5


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

    # Build the training model with sequential model
    inputs = Input(shape = (9,), name = 'input_layer')

    hidden1 = Dense(300,
                    kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
                    activation = 'relu',
                    name = 'hidden_layer_1')(inputs)

    hidden2 = Dense(300,
                    kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
                    activation = 'relu',
                    name = 'hidden_layer_2')(hidden1)

    hidden3 = Dense(100,
                    kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
                    activation = 'relu',
                    name = 'hidden_layer_3')(hidden2)

    outputs = Dense(1,
                    kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
                    name = 'output_layer')(hidden2)

    feed_forward_NN = Model(inputs = inputs,
                            outputs = outputs,
                            name = 'feed_forward_NN')

    # Print the NN architecture and the NN topology
    feed_forward_NN.summary()
    keras.utils.plot_model(feed_forward_NN, to_file = 'NN_architecture.pdf', show_shapes = True,
                           show_layer_activations = True)

    # Configure the training parameters
    feed_forward_NN.compile(optimizer = keras.optimizers.Adamax(beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
                            loss = tf.losses.MeanSquaredError(),
                            metrics = ['mae'],
                            run_eagerly = False)

    # Start of training process
    print("Fit model on training data")

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = feed_forward_NN.fit(X_train,
                                  t_train,
                                  epochs = 1000,
                                  verbose = 2,
                                  batch_size = 512,  # 64 best
                                  validation_data = (X_val, t_val),
                                  validation_batch_size = 32,
                                  callbacks = [callback]
                                  )

    # Evaluate the model with test dataset
    print("\n Evaluate on test data")

    results = feed_forward_NN.evaluate(X_test, t_test, batch_size = 32)

    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(mae) + 1)

    print("test loss: {}, test mae: {}".format(results[0], results[1]))

    # End the clock
    end = time.perf_counter()
    print("\n runtime: %s second " % (end - start))

    # Prediction in test dataset
    print("\n Generate predictions")

    # interval sampling from test dataset
    X_pred = X_test[::NUM_TEST_SAMPLE_POINTS]
    t_pred_true = t_test[::NUM_TEST_SAMPLE_POINTS]
    t_pred = feed_forward_NN.predict(X_pred, batch_size = 32)

    # interval sampling from train dataset
    X_pred_train = X_train[::NUM_TEST_SAMPLE_POINTS]
    t_pred_true_train = t_train[::NUM_TEST_SAMPLE_POINTS]
    t_pred_train = feed_forward_NN.predict(X_pred_train,
                                           batch_size = 32)

    # interval sampling from val dataset
    X_pred_val = X_val[::NUM_TEST_SAMPLE_POINTS]
    t_pred_true_val = t_val[::NUM_TEST_SAMPLE_POINTS]
    t_pred_val = feed_forward_NN.predict(X_pred_val,
                                         batch_size = 32)

    # interval sampling from all dataset
    X_pred_all = x[::NUM_TEST_SAMPLE_POINTS]
    t_pred_true_all = t[::NUM_TEST_SAMPLE_POINTS]
    t_pred_all = feed_forward_NN.predict(x,
                                         batch_size = 32)

    # The mean error of regression
    mean_error = abs(np.mean(t_pred - t_pred_true))
    mean_error_train = abs(np.mean(t_pred_train - t_pred_true_train))
    mean_error_val = abs(np.mean(t_pred_val - t_pred_true_val))
    mean_error_mean = abs(np.mean([mean_error, mean_error_val, mean_error_val]))

    print("number of sample points: {}\n".format(str(len(X_pred))),
          "prediction shape: {} target shape: {}".format(str(t_pred.shape), str(t_pred_true.shape)))

    print("\nmean error of regression:", mean_error)

    # Statistic analyse: calculate relationship coefficient R and determination coefficient R^2
    # Test dataset
    R_2 = sklearn.metrics.r2_score(t_pred_true, t_pred)
    R = R_2 ** 0.5

    # Train dataset
    R_2_train = sklearn.metrics.r2_score(t_pred_true_train, t_pred_train)
    R_train = R_2_train ** 0.5

    # Val dataset
    R_2_val = sklearn.metrics.r2_score(t_pred_true_val, t_pred_val)
    R_val = R_2_val ** 0.5

    # All dataset
    R_2_all = sklearn.metrics.r2_score(t_pred_true_all, t_pred_all)
    R_all = R_2_all ** 0.5

    # Statistic analyse: linear regression with "The Least Squares"
    # Test dataset
    t_pred_true_reshape = np.reshape(t_pred_true[:len(t_pred_true)], (len(t_pred_true),))
    t_pred_reshape = np.reshape(t_pred[:len(t_pred)], (len(t_pred),))

    Target = np.vstack([t_pred_true_reshape,
                        np.ones(len(t_pred_true_reshape))
                        ]).T
    Output = t_pred_reshape
    beta1, beta2 = np.linalg.lstsq(Target, Output, rcond = None)[0]

    # Train dataset
    t_pred_true_train_reshape = np.reshape(t_pred_true_train[:len(t_pred_true_train)], (len(t_pred_true_train),))
    t_pred_train_reshape = np.reshape(t_pred_train[:len(t_pred_train)], (len(t_pred_train),))

    Target_train = np.vstack([t_pred_true_train_reshape,
                              np.ones(len(t_pred_true_train_reshape))
                              ]).T
    Output_train = t_pred_train_reshape
    beta1_train, beta2_train = np.linalg.lstsq(Target_train, Output_train, rcond = None)[0]

    # Val dataset
    t_pred_true_val_reshape = np.reshape(t_pred_true_val[:len(t_pred_true_val)], (len(t_pred_true_val),))
    t_pred_val_reshape = np.reshape(t_pred_val[:len(t_pred_val)], (len(t_pred_val),))

    Target_val = np.vstack([t_pred_true_val_reshape,
                            np.ones(len(t_pred_true_val_reshape))
                            ]).T
    Output_val = t_pred_val_reshape
    beta1_val, beta2_val = np.linalg.lstsq(Target_val, Output_val, rcond = None)[0]

    # all dataset
    t_pred_true_all_reshape = np.reshape(t_pred_true_all[:len(t_pred_true_all)], (len(t_pred_true_all),))
    t_pred_all_reshape = np.reshape(t_pred_all[:len(t_pred_all)], (len(t_pred_all),))

    Target_all = np.vstack([t_pred_true_all_reshape,
                            np.ones(len(t_pred_true_all_reshape))
                            ]).T
    Output_all = t_pred_all_reshape
    beta1_all, beta2_all = np.linalg.lstsq(Target_all, Output_all, rcond = None)[0]

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
    plt.savefig('convergence_history.pdf', format = 'pdf', bbox_inches = 'tight')
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
    plt.savefig('training_accuracy.pdf', format = 'pdf', bbox_inches = 'tight')
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
    plt.savefig('Prediction_vs_Target.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()

    # Error histogram
    plt.figure('Error Histogram', tight_layout = True)
    plt.title('Error Histogram with 20 Bins')
    plt.xlabel('Errors = Targets - Outputs')
    plt.ylabel('Instances')
    plt.xlim(- mean_error_mean * 50, mean_error_mean * 50)
    plt.ylim(0, 2e5)
    plt.ticklabel_format(style = 'sci', scilimits = (-1, 5), axis = 'y', useMathText = True)
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
    plt.xticks(np.linspace(- mean_error_mean * 50, mean_error_mean * 50, 20), rotation = 90)

    n1, bins1, patches1 = plt.hist(t_pred_train - t_pred_true_train,
                                   np.linspace(- mean_error_mean * 50, mean_error_mean * 50, 20),
                                   histtype = 'bar',
                                   bottom = 0,
                                   align = 'mid',
                                   rwidth = 0.8,
                                   facecolor = 'blue',
                                   stacked = False,
                                   label = 'Training Error')

    height_train = patches1.datavalues

    n2, bins2, patches2 = plt.hist(t_pred_val - t_pred_true_val,
                                   np.linspace(- mean_error_mean * 50, mean_error_mean * 50, 20),
                                   histtype = 'bar',
                                   bottom = height_train,
                                   align = 'mid',
                                   rwidth = 0.8,
                                   facecolor = 'green',
                                   stacked = False,
                                   label = 'Validation Error')

    height_val = patches2.datavalues

    n3, bins3, patches3 = plt.hist(t_pred - t_pred_true,
                                   np.linspace(- mean_error_mean * 50, mean_error_mean * 50, 20),
                                   histtype = 'bar',
                                   bottom = height_train + height_val,
                                   align = 'mid',
                                   rwidth = 0.8,
                                   facecolor = 'red',
                                   stacked = False,
                                   label = 'Test Error')

    # Zero error line
    plt.axvline(x = 0, ymin = 0, ymax = 1, color = 'orange', label = 'Zero Error')
    plt.legend()

    # Save figure：Error histogram
    plt.savefig('Error_Histogram.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()

    # fitted curvature versus the exact curvature in different dataset
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 6))

    # reference line
    x_ref = np.linspace(-2.5, 2.5, 10)
    y_ref = x_ref

    # predict vs target: Training data set
    axs[0, 0].scatter(t_pred_true_train, t_pred_train, marker = 'o', c = 'white', linewidths = 1, edgecolors = 'black',
                      label = 'Data')
    axs[0, 0].set_title('Train: R=')
    axs[0, 0].text(0.068, 0.27, '%.5f' % R_train, fontsize = 12)
    axs[0, 0].set_xlabel('Target')
    axs[0, 0].set_ylabel('Prediction')
    axs[0, 0].set_xlim(-0.25, 0.25)
    axs[0, 0].set_ylim(-0.25, 0.25)
    axs[0, 0].plot(x_ref, y_ref, color = 'black', linestyle = '--', label = 'Y=T')
    axs[0, 0].plot(x_ref, beta1_train * x_ref + beta2_train, color = 'blue', label = 'Fit')
    axs[0, 0].legend()

    # predict vs target: Validation set
    axs[0, 1].scatter(t_pred_true_val, t_pred_val, marker = 'o', c = 'white', linewidths = 1, edgecolors = 'black',
                      label = 'Data')
    axs[0, 1].set_title('Validation: R=')
    axs[0, 1].text(0.102, 0.27, '%.5f' % R_val, fontsize = 12)
    axs[0, 1].set_xlabel('Target')
    axs[0, 1].set_ylabel('Prediction')
    axs[0, 1].set_xlim(-0.25, 0.25)
    axs[0, 1].set_ylim(-0.25, 0.25)
    axs[0, 1].plot(x_ref, y_ref, color = 'black', linestyle = '--', label = 'Y=T')
    axs[0, 1].plot(x_ref, beta1_val * x_ref + beta2_val, color = 'green', label = 'Fit')
    axs[0, 1].legend()

    # predict vs target: Test data set
    axs[1, 0].scatter(t_pred_true, t_pred, marker = 'o', c = 'white', linewidths = 1, edgecolors = 'black',
                      label = 'Data')
    axs[1, 0].set_title('Test: R=')
    axs[1, 0].text(0.06, 0.268, '%.5f' % R, fontsize = 12)
    axs[1, 0].set_xlabel('Target')
    axs[1, 0].set_ylabel('Prediction')
    axs[1, 0].set_xlim(-0.25, 0.25)
    axs[1, 0].set_ylim(-0.25, 0.25)
    axs[1, 0].plot(x_ref, y_ref, color = 'black', linestyle = '--', label = 'Y=T')
    axs[1, 0].plot(x_ref, beta1 * x_ref + beta2, color = 'red', label = 'Fit')
    axs[1, 0].legend()

    # predict vs target: all data set
    axs[1, 1].scatter(t_pred_true_all, t_pred_all, marker = 'o', c = 'white', linewidths = 1, edgecolors = 'black',
                      label = 'Data')
    axs[1, 1].set_title('All: R=')
    axs[1, 1].text(0.05, 0.268, '%.5f' % R_all, fontsize = 12)
    axs[1, 1].set_xlabel('Target')
    axs[1, 1].set_ylabel('Prediction')
    axs[1, 1].set_xlim(-0.25, 0.25)
    axs[1, 1].set_ylim(-0.25, 0.25)
    axs[1, 1].plot(x_ref, y_ref, color = 'black', linestyle = '--', label = 'Y=T')
    axs[1, 1].plot(x_ref, beta1_all * x_ref + beta2_all, color = 'gray', label = 'Fit')
    axs[1, 1].legend()

    fig.text(0.52, 0.6, regress_equ(beta1_train, beta2_train), rotation = 90, fontsize = 8)
    fig.text(0.02, 0.6, regress_equ(beta1_val, beta2_val), rotation = 90, fontsize = 8)
    fig.text(0.02, 0.1, regress_equ(beta1, beta2), rotation = 90, fontsize = 8)
    fig.text(0.52, 0.1, regress_equ(beta1_all, beta2_all), rotation = 90, fontsize = 8)

    plt.tight_layout()
    plt.subplots_adjust(left = 0.12, wspace = 0.4)
    plt.show()

    # Save figure：predict vs target with all data
    plt.savefig('Fitted_Curvature_vs_Exact_Curvature.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()

    # End of process
    print("process finish")