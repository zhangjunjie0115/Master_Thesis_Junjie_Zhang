"""
ML-Train: A code to optimize the hyperparameters of the simple Convolutional Neural Network with optuna framework

Visualization Steps:
1. generate the data-bank of sqlite
2. open terminal and use " source activate tensorflow " to activate the environment from base to tensorflow
3. tap the command " optuna-dashboard sqlite:/// [data-bank name].sqlite3 "
4. open browser and tap ports to check the dashboard
** Database naming conflict will cause loading failure **

features:
1. 10 Hyperparameters: Lr, Optimizer, Train batch size, dense neurons, activation function,
                      Strides, kernel numbers, kernel size
                      Structure: only LeNet
2. Pruning hook for val_loss
3. Fix Multi ThreadPool method
4. Add opt_LearningRateSchedule
5. Save results as .pdf file

"""

import tensorflow as tf
import matplotlib
import scipy.io
import sklearn
import optuna
from tensorflow import keras
from keras import models
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# load the generated data and obtain the values for x, t:
data = scipy.io.loadmat('Data/database.mat')  # Import dataset
x = data["datafra"]  # 188232 x 9 double： fraction of volume
t = data["datacur"]  # 188232 x 1 double:  generated curvature

# Repacking and reshape VOF-Matrix x as a 3x3 Matrix:
x1, x2, x3, x4, x5, x6, x7, x8, x9 = tf.split(x, num_or_size_splits = 9, axis = 1)
x_h1 = tf.concat([x7, x8, x9], axis = 1)
x_h2 = tf.concat([x4, x5, x6], axis = 1)
x_h3 = tf.concat([x1, x2, x3], axis = 1)
x_reshape = tf.stack([x_h1, x_h2, x_h3])
x = np.reshape(np.array(x), (188232, 3, 3, 1))

# Process the raw data
# split the dataset to 70% Training set, 15% test set and 15% validation set
X_train, X_test, t_train, t_test = train_test_split(x, t, test_size = 0.3, random_state = 321)
X_test, X_val, t_test, t_val = train_test_split(X_test, t_test, test_size = 0.5, random_state = 321)
disp_Datashape = [["Input data", str(x.shape)], ["Output data", str(t.shape)],
                  ["X_Train", str(X_train.shape)], ["t_Train", str(t_train.shape)],
                  ["X_Test", str(X_test.shape)], ["t_test", str(t_test.shape)],
                  ["X_Val", str(X_val.shape)], ["t_val", str(t_val.shape)]]

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


def objective(trial):
    # Generate search space for hyperparameters
    opt_structure = trial.suggest_categorical("Structure", ["LeNet"])

    opt_optimizer = trial.suggest_categorical("Optimizer", ["Adam", "Adamax", "Nadam", "Adadelta", "Ftrl"])

    opt_lr_method = trial.suggest_categorical("Lr_method", ["LearningRateSchedule", "FixedLearningRate"])

    opt_lr = trial.suggest_float("Learning_Rate",
                                 low = 1e-6,
                                 high = 1e-2,
                                 log = True)

    opt_units = trial.suggest_int("Units",
                                  low = 50,
                                  high = 300,
                                  step = 50)

    opt_train_batch = trial.suggest_categorical("Train_batch", [32, 64, 128, 256, 512])

    opt_conv_filters = trial.suggest_categorical("Kernel_numbers", [8, 16, 32, 64, 128])

    opt_kernel_size = trial.suggest_categorical("Kernel_size", [(1, 1), (2, 2), (3, 3)])

    opt_strides = trial.suggest_categorical("Strides", [(1, 1), (2, 2)])

    opt_activation_fn = trial.suggest_categorical("Activation",
                                                  ["relu", "selu", "elu"])  # worse results with tanh

    def scheduler(epoch, lr):
        """
        config learning rate scheduler
        Args:
            lr: learning rate
            epoch: training epoch

        Returns: changeable learning rate or fixed learning rate

        """
        if opt_lr_method == "LearningRateSchedule":

            if epoch < 20:
                lr = 3e-4
            elif epoch < 40:
                lr = 9e-5
            elif epoch < 60:
                lr = 7e-5
            elif epoch < 80:
                lr = 5e-5
            else:
                lr = 3e-5

        elif opt_lr_method == "FixedLearningRate":

            lr = opt_lr

        return lr

    # build the training model with sequential model
    if opt_structure == "LeNet":

        # Build the LeNet training model with sequential model
        inputs = Input(shape = (3, 3, 1), name = 'input_layer')

        norm = layers.Normalization(mean = 0., variance = 1.)(inputs)

        conv1 = layers.Conv2D(
            filters = opt_conv_filters,
            kernel_size = opt_kernel_size,
            strides = (1, 1),
            padding = "same",
            activation = 'relu',
            kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
            name = 'convolution_layer_1')(norm)

        pooling1 = layers.MaxPooling2D(
            pool_size = (1, 1),
            strides = (1, 1),
            name = 'Max_pooling1')(conv1)

        flatten = layers.Flatten(name = 'Flatten')(pooling2)

        hidden = Dense(units = opt_units,
                       kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
                       activation = opt_activation_fn,
                       name = 'hidden_layer')(flatten)
        outputs = Dense(1,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 123),
                        name = 'output_layer')(hidden)

    elif opt_structure == "2_Hidden_Layers":
        inputs = Input(shape = (9,), name = 'input_layer')

        hidden1 = Dense(units = opt_units,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(),
                        activation = opt_activation_fn,
                        name = 'hidden_layer_1')(inputs)

        hidden2 = Dense(units = opt_units,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(),
                        activation = opt_activation_fn,
                        name = 'hidden_layer_2')(hidden1)

        outputs = Dense(1,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(),
                        name = 'output_layer')(hidden2)

    else:
        inputs = Input(shape = (9,), name = 'input_layer')

        hidden1 = Dense(units = opt_units,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(),
                        activation = opt_activation_fn,
                        name = 'hidden_layer_1')(inputs)

        hidden2 = Dense(units = opt_units,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(),
                        activation = opt_activation_fn,
                        name = 'hidden_layer_2')(hidden1)

        hidden3 = Dense(units = opt_units,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(),
                        activation = opt_activation_fn,
                        name = 'hidden_layer_3')(hidden2)

        outputs = Dense(1,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(),
                        name = 'output_layer')(hidden3)

    CNN = Model(inputs = inputs,
                outputs = outputs,
                name = 'CNN')

    # print the CNN architecture
    CNN.summary()

    CNN.compile(optimizer = opt_optimizer,
                loss = tf.losses.MeanSquaredError(),
                metrics = ['mae'],
                run_eagerly = False)

    # Compile training model
    # Configure the training method
    # Performance in train process: Whether the trial should be pruned
    # Using "val_loss" for pruning bed trails
    print("Fit model on training data")
    history = CNN.fit(X_train,
                      t_train,
                      epochs = 1000,
                      verbose = 0,
                      batch_size = opt_train_batch,
                      validation_data = (X_val, t_val),
                      validation_batch_size = 32,
                      callbacks = [optuna.integration.TFKerasPruningCallback(trial, "val_loss"),
                                   tf.keras.callbacks.LearningRateScheduler(scheduler)
                                   ]
                      )

    # Evaluate the model with test dataset
    print("Evaluate on test data")
    results = CNN.evaluate(X_test, t_test, batch_size = 32)

    # Using test loss as the indicator for optimization
    score = results
    return score[0]


if __name__ == "__main__":

    # Create a new study
    study = 'CNN_opt_study_v1'
    study = optuna.create_study(study_name = study,
                                direction = "minimize",
                                sampler = optuna.samplers.TPESampler(seed = 123),
                                pruner = optuna.pruners.MedianPruner(
                                    n_startup_trials = 5,
                                    n_warmup_steps = 100,
                                    interval_steps = 100),
                                storage = "sqlite:///CNN_v1.sqlite3")

    with ThreadPoolExecutor(max_workers = 5) as executor:
        for _ in range(5):
            executor.submit(study.optimize, objective, n_trials = 20)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Results of the study
    print("Study statistics: ")
    print(" Number of finished trials: {}".format(len(study.trials)))
    print(" Number of pruned trials: {}".format(len(pruned_trials)))
    print(" Number of complete trials: {}".format(len(complete_trials)))

    # Parameters of the best trial
    print(" Best trial: ")
    trial = study.best_trial
    print(" Value: {}".format(trial.value))

    print(" Parameters: ")
    for key, value in trial.params.items():
        print("   {}:  {}".format(key, value))

    # Visualization the study results
    optuna.visualization.matplotlib.plot_optimization_history(study)
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('optimization_history_cnn.pdf', format = 'pdf', bbox_inches = 'tight')

    optuna.visualization.matplotlib.plot_param_importances(study)
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('param_importance_cnn.pdf', format = 'pdf', bbox_inches = 'tight')

    optuna.visualization.matplotlib.plot_intermediate_values(study)
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('intermediate_values_cnn.pdf', format = 'pdf', bbox_inches = 'tight')
