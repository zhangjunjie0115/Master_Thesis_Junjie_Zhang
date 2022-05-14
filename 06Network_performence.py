"""
ML Model evaluate: A code to evaluate the CNN(LeNet) with K-Fold-CV

features:
1. dataset split module
2. extensible sequential keras module
3. Learning Rate Schedule + ReduceLROnPlateau
4. statistical analysis module and regression function
5. visualization module
6. sampling module for test dataset
7. function of "repacking and reshape VOF-Matrix"
8. time clock of program running. main program --> end of evaluate
9. New Repacking and Reshape Algorithms
10. Error histogram for trained model in different dataset
11. Predict vs Target for all data
12. CVSLog callback for recording training history
13. tensorboard callback for monitoring training state. Open a Terminal at the parent dictionary， then tap "tensorboard --logdir=<LOG_PATH_NAME>"
14. k-fold cross validation

** results documents should be .PDF instead of .PNG
** padding: "same": dim won't change; "valid": dim decrease 1

"""

import time
import matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import sklearn
from tensorflow import keras
from keras import (models, layers)
from keras.layers import (Input, Dense)
from keras.models import Model
from sklearn.model_selection import (train_test_split, KFold)

# Global variable
# Number of iterations of k-fold cross validation
K_FOLD_SPLIT = 5
# Sample interval in all split set for prediction. It will change the statistic values for prediction and the density of
# points in  the figure
NUM_TEST_SAMPLE_POINTS = 1


def LeNet():
    # Build a standard CNN(LeNet) training model with Keras sequential model
    inputs = Input(shape = (3, 3, 1),
                   batch_size = None,
                   name = 'input_layer')

    # three types of normalization. Performance for this case: Normalization > batch normalization(BN)
    # layer normalization doesn't adapt to this case
    norm = layers.Normalization(mean = 0., variance = 1.)(inputs)

    batchnorm = layers.BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True)(
        inputs)

    layernorm = layers.LayerNormalization(axis = 1, epsilon = 0.001, center = True, scale = True)(inputs)

    # Best performance: three convolutional layer with one Max-pooling layer
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

    pooling3 = layers.MaxPooling2D(
        pool_size = (1, 1),
        strides = (1, 1),
        padding = "valid",
        name = 'Max_pooling3')(conv3)

    flatten = layers.Flatten(name = 'Flatten')(pooling3)

    hidden1 = Dense(100,
                    kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
                    activation = 'relu',
                    name = 'hidden_layer_1')(flatten)

    outputs = Dense(1,
                    kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
                    name = 'output_layer')(hidden1)

    LeNet = Model(inputs = inputs,
                  outputs = outputs,
                  name = 'LeNet')

    return LeNet


def CNNLSTM():
    # build a CNN architecture and using LSTM units to replace dense layer, need a virtual time axis for LSTM.
    # LSTM can reduce the number of trainable parameters meanwhile keeps a similar performance as standard CNN
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
        units = 20,
        activation = 'tanh',
        recurrent_activation = 'relu',
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
        dropout = 0.0,
        recurrent_dropout = 0.0,
    )(dim_expand)

    # Regression output layer
    outputs = Dense(1,
                    kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
                    name = 'output_layer')(LSTM)

    CNNLSTM = Model(inputs = inputs,
                    outputs = outputs,
                    name = 'CNNLSTM')

    return CNNLSTM


if __name__ == "__main__":

    # Start the clock
    start = time.perf_counter()

    # Load the generated matlab data and obtain the values for x, t:
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

    # K-fold cross validation and save the best model
    cross_validation_loss_score = []
    cross_validation_mae_score = []

    Model_Checkpoint_dir = 'Model Checkpoint/'
    fold_var = 1

    for train, test in KFold(n_splits = K_FOLD_SPLIT, shuffle = True, random_state = 321).split(x, t):

        print(("\n the {} fold cross validation".format(fold_var)))

        # Choose framework for CNN
        model = LeNet()

        # Print the NN architecture and the NN topology
        model.summary()
        keras.utils.plot_model(model, to_file = 'CNN_architecture.pdf', show_shapes = True,
                               show_layer_activations = True)

        # Configure the training parameters
        model.compile(
            optimizer = keras.optimizers.Adamax(learning_rate = 3e-4, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
            # keras.optimizers.Adam(beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8),
            loss = tf.losses.MeanSquaredError(),
            metrics = ['mae'],
            run_eagerly = False)

        # Start of training process
        print("Fit model on training data")

        history = model.fit(x[train],
                            t[train],
                            epochs = 10,
                            verbose = 2,
                            batch_size = 6400,
                            validation_split = 0.2,
                            validation_batch_size = 32,
                            callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                                              factor = 0.9,
                                                                              patience = 20,
                                                                              verbose = 1,
                                                                              mode = 'auto',
                                                                              min_delta = 1e-8,
                                                                              cooldown = 20,
                                                                              min_lr = 3e-10, ),

                                         tf.keras.callbacks.CSVLogger('train_csv/log_cnn.csv', separator = ',',
                                                                      append = True),

                                         tf.keras.callbacks.TensorBoard(log_dir = 'tensorboard logs',
                                                                        histogram_freq = 10,
                                                                        write_graph = True,
                                                                        write_images = True,
                                                                        update_freq = 10,
                                                                        embeddings_freq = 10,
                                                                        ),

                                         tf.keras.callbacks.ModelCheckpoint(Model_Checkpoint_dir + 'best_model'+str(fold_var)+'.h5',
                                                                            save_weights_only = False,
                                                                            save_best_only = True,
                                                                            monitor = 'val_loss',
                                                                            verbose=1,
                                                                            mode = 'min',
                                                                            ),
                                         ]
                            )

        # Evaluate the model with test dataset
        print("\n Evaluate on test data")

        # Load best model to evaluate the performance of the model
        if os.path.exists(Model_Checkpoint_dir + 'best_model'+str(fold_var)+'.h5'):
            model.load_weights(Model_Checkpoint_dir + 'best_model'+str(fold_var)+'.h5')
            print("best weight of {} fold loaded".format(str(fold_var)))

        results = model.evaluate(x[test],
                                 t[test],
                                 batch_size = 32,
                                 verbose = 0,
                                 callbacks = None
                                 )
        print("test loss: {}, test mae: {}".format(results[0], results[1]))

        cross_validation_loss_score.append(results[0])
        cross_validation_mae_score.append(results[1])

        fold_var += 1

    print("k-fold test loss: {} (+/- {})".format(np.mean(cross_validation_loss_score),
                                                 np.std(cross_validation_loss_score)))
    print("\n k-fold test mae: {} (+/- {})".format(np.mean(cross_validation_mae_score),
                                                   np.std(cross_validation_mae_score)))

    # End of k-fold cross validation

    # End of process
    print("process finish")
