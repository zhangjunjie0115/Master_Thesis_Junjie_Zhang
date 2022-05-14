"""
ML-Train: A code to optimize the hyperparameters of the Feed forward Neural Network with optuna framework

Visualization Steps:
1. generate the data-bank of sqlite
2. open terminal and use " source activate tensorflow " to activate the environment from base to tensorflow
3. tap the command " optuna-dashboard sqlite:/// [data-bank name].sqlite3 "
4. open browser and tap ports to check the dashboard
** Database naming conflict will cause loading failure **

features:
1. 6 Hyperparameters: Lr, Optimizer, Train batch size, neurons, hidden layers, activation function
2. Pruning hook for val_loss
3. Fix Multi ThreadPool method
4. Add opt_LearningRateSchedule

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
    opt_structure = trial.suggest_categorical("Structure",
                                              ["1_Hidden_Layer", "2_Hidden_Layers", "3_Hidden_Layers"])

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
    if opt_structure == "1_Hidden_Layer":
        inputs = Input(shape = (9,), name = 'input_layer')

        hidden1 = Dense(units = opt_units,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(),
                        activation = opt_activation_fn,
                        name = 'hidden_layer_1')(inputs)

        outputs = Dense(1,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(),
                        name = 'output_layer')(hidden1)

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

    feed_forward_NN = Model(inputs = inputs,
                            outputs = outputs,
                            name = 'feed_forward_NN')

    # print the NN architecture
    feed_forward_NN.summary()

    feed_forward_NN.compile(optimizer = opt_optimizer,
                            loss = tf.losses.MeanSquaredError(),
                            metrics = ['mae'],
                            run_eagerly = False)

    # Compile training model
    # Configure the training method
    # Performance in train process: Whether the trial should be pruned
    print("Fit model on training data")
    history = feed_forward_NN.fit(X_train,
                                  t_train,
                                  epochs = 1000,
                                  verbose = 0,
                                  batch_size = opt_train_batch,
                                  validation_data = (X_val, t_val),
                                  validation_batch_size = 32,
                                  callbacks = [optuna.integration.TFKerasPruningCallback(trial, "val_mae"),
                                               tf.keras.callbacks.LearningRateScheduler(scheduler)
                                               ]
                                  )

    # Evaluate the model with test dataset
    print("Evaluate on test data")
    results = feed_forward_NN.evaluate(X_test, t_test, batch_size = 32)

    # Mark of optimization
    score = results
    return score[0]


if __name__ == "__main__":

    study = 'feed_forward_NN_opt_study_v8'
    study = optuna.create_study(study_name = study,
                                direction = "minimize",
                                sampler = optuna.samplers.TPESampler(seed = 123),
                                pruner = optuna.pruners.MedianPruner(
                                    n_startup_trials = 5,
                                    n_warmup_steps = 100,
                                    interval_steps = 100),
                                storage = "sqlite:///opt_BPNN_v8.sqlite3")

    with ThreadPoolExecutor(max_workers = 5) as executor:
        for _ in range(5):
            executor.submit(study.optimize, objective, n_trials = 20)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print(" Number of finished trials: {}".format(len(study.trials)))
    print(" Number of pruned trials: {}".format(len(pruned_trials)))
    print(" Number of complete trials: {}".format(len(complete_trials)))

    print(" Best trial: ")
    trial = study.best_trial
    print(" Value: {}".format(trial.value))

    print(" Parameters: ")
    for key, value in trial.params.items():
        print("   {}:  {}".format(key, value))

    # Visualization the study results
    optuna.visualization.matplotlib.plot_optimization_history(study)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

    optuna.visualization.matplotlib.plot_param_importances(study)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

    optuna.visualization.matplotlib.plot_intermediate_values(study)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()
