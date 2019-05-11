import argparse
import pandas as pd
import seaborn as sn
from sklearn import neural_network
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metric
import numpy as np
import matplotlib.pyplot as plt

from methods.MLP import MLP

FRAC_VALIDATION = 0.2

parser = argparse.ArgumentParser(description='Neural Networks.')
parser.add_argument('-training', dest='training_path')
parser.add_argument('-test', dest='test_path')


def getParameters(n_features):

    val = input('Set number of hidden layers (default: 1): ')
    num_hidden = 1
    if val != '':
        num_hidden = int(val)

    nn_hidden = []
    for i in range(num_hidden):
        val = input('Set number of neurons in hidden layer ' + str(i+1) + ' (default: ' + str(int(n_features/2)) + '): ')
        nn_hidden.append(int(n_features/2))
        if val != '':
            nn_hidden[i] = int(val)

    val = input('Set number of classes: ')
    n_classes = 2
    if val != '':
        n_classes = int(val)

    # Set Neural Network Architecture
    nn_arch = [n_features]
    for i in range(len(nn_hidden)):
        nn_arch.append(nn_hidden[i])
    nn_arch.append(n_classes)

    val = input('Set activation function: (default: "logistic"): ')
    activation = 'logistic'
    if val != '':
        activation = val

    val = input('Set learning rate: (default: 0.0001): ')
    eta = 0.0001
    if val != '':
        eta = float(val)

    val = input('Set number of epochs: (default: 1000): ')
    epochs = 1000
    if val != '':
        epochs = int(val)

    val = input('Normalize the input data: (default: "no"): ')
    normalize = 'no'
    if val != '':
        normalize = val

    parameters = {'architecture': nn_arch,
                  'activation': activation,
                  'eta': eta,
                  'epochs': epochs,
                  'normalize': normalize}

    return parameters


def multilayer_perceptron(train_input, train_label, val_input, val_label, test_input, test_label):
    print("Starting Multilayer Perceptron...")

    # Get Parameters
    params = getParameters(train_input.shape[1])

    # Normalize the inputs
    if params['normalize'] == 'yes':
        train_input = train_input.astype('float32')/255.0
        val_input = val_input.astype('float32')/255.0

    # Train the model
    model = MLP.build_model(train_input, train_label, params['architecture'], params['activation'], 0.1,
                            params['eta'], params['epochs'])

    print('Parameters: ', params)

    # Plot Training Error
    fig = plt.figure()
    plt.plot(model['cost'])
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
    fig.savefig('error_plot.png')

    # Predict labels from validation
    print('Validation:')
    y_pred = MLP.predict(val_input, model)

    # Plot metrics
    accuracy = metric.accuracy_score(np.array(val_label).flatten(), np.array(y_pred).flatten(), normalize=True)
    print('- Accuracy: ', accuracy)  # show accuracy score
    print('- Precision: ', metric.precision_score(val_label, y_pred, average='macro'))  # show precision
    print('- Recall: ', metric.recall_score(val_label, y_pred, average='macro'))    # show recall
    print('- F1-score: ', metric.f1_score(val_label, y_pred, average='macro'))  # show F1 score

    # Plot Confusion matrix
    cm = confusion_matrix(val_label, y_pred)
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]   # normalize
    df_cm = pd.DataFrame(cm, index=[i for i in "0123456789"], columns=[i for i in "0123456789"])
    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    fig.savefig('cm_validation.png')

    val = input('Predict Test Set: (default: "no"): ')
    test = 'no'
    if val != '':
        test = val

    # Predict labels from test
    if test == 'yes':
        print('Test:')
        y_pred = MLP.predict(test_input, model)

        # Plot metrics
        accuracy = metric.accuracy_score(np.array(test_label).flatten(), np.array(y_pred).flatten(), normalize=True)
        print('- Accuracy: ', accuracy)  # show accuracy score
        print('- Precision: ', metric.precision_score(test_label, y_pred, average='macro'))  # show precision
        print('- Recall: ', metric.recall_score(test_label, y_pred, average='macro'))  # show recall
        print('- F1-score: ', metric.f1_score(test_label, y_pred, average='macro'))  # show F1 score

        # Plot Confusion matrix
        cm = confusion_matrix(test_label, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize
        df_cm = pd.DataFrame(cm, index=[i for i in "0123456789"], columns=[i for i in "0123456789"])
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show()
        fig.savefig('cm_test.png')


def scikit_multilayer_perceptron(train_input, train_label, val_input, val_label, test_input, test_label):
    print("Starting Scikit Multilayer Preceptron...")

    # Get Parameters
    params = getParameters(train_input.shape[1])
    print('Parameters: ', params)

    # Normalize the inputs
    if params['normalize'] == 'yes':
        train_input = train_input.astype('float32')/255.0
        val_input = val_input.astype('float32')/255.0

    # Train the model
    model = neural_network.MLPClassifier(hidden_layer_sizes=tuple(params['architecture'][1:-1]),
                                         activation=params['activation'], max_iter=params['epochs'],
                                         learning_rate_init=params['eta'])
    model.fit(train_input, train_label)

    # Predict labels from validation
    print('Validation:')
    y_pred = model.predict(val_input)

    # Plot metrics
    accuracy = metric.accuracy_score(np.array(val_label).flatten(), np.array(y_pred).flatten(), normalize=True)
    print('- Accuracy: ', accuracy)  # show accuracy score
    print('- Precision: ', metric.precision_score(val_label, y_pred, average='macro'))  # show precision
    print('- Recall: ', metric.recall_score(val_label, y_pred, average='macro'))  # show recall
    print('- F1-score: ', metric.f1_score(val_label, y_pred, average='macro'))  # show F1 score

    # Plot Confusion matrix
    cm = confusion_matrix(val_label, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize
    df_cm = pd.DataFrame(cm, index=[i for i in "0123456789"], columns=[i for i in "0123456789"])
    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    fig.savefig('cm_validation.png')

    val = input('Predict Test Set: (default: "no"): ')
    test = 'no'
    if val != '':
        test = val

    # Predict labels from test
    if test == 'yes':
        print('Test:')
        y_pred = model.predict(test_input)

        # Plot accuracy
        accuracy = metric.accuracy_score(np.array(test_label).flatten(), np.array(y_pred).flatten(), normalize=True)
        print('- Accuracy: ', accuracy)  # show accuracy score
        print('- Precision: ', metric.precision_score(test_label, y_pred, average='macro'))  # show precision
        print('- Recall: ', metric.recall_score(test_label, y_pred, average='macro'))  # show recall
        print('- F1-score: ', metric.f1_score(test_label, y_pred, average='macro'))  # show F1 score

        # Plot Confusion matrix
        cm = confusion_matrix(test_label, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize
        df_cm = pd.DataFrame(cm, index=[i for i in "0123456789"], columns=[i for i in "0123456789"])
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show()
        fig.savefig('cm_test.png')


def init_dataset(training_path, test_path):
    print("Initializing dataset...")

    # Read training dataset
    df_train = pd.read_csv(training_path)
    df_test = pd.read_csv(test_path)

    # Split training data in training and validation
    validation_set = df_train.sample(frac=FRAC_VALIDATION, random_state=1)
    training_set = df_train.drop(validation_set.index)

    print('Training set dimensions (', (1 - FRAC_VALIDATION) * 100.0, '% ):', training_set.shape)
    print('Validation set dimensions (', FRAC_VALIDATION * 100.0, '% ):', validation_set.shape)

    # Split training set in input and label
    train_input = training_set.iloc[:, 1:training_set.shape[1]].values
    train_label = training_set.iloc[:, 0].values

    # Split validation set in input and label
    val_input = validation_set.iloc[:, 1:validation_set.shape[1]].values
    val_label = validation_set.iloc[:, 0].values

    # Split test set in input and label
    test_input = df_test.iloc[:, 1:df_test.shape[1]].values
    test_label = df_test.iloc[:, 0].values

    return train_input, train_label, val_input, val_label, test_input, test_label


def main():
    args = parser.parse_args()

    # Load datasets
    train_input, train_label, val_input, val_label, test_input, test_label = init_dataset(args.training_path,
                                                                                          args.test_path)

    print('Choose your method:')
    print('1 - Multilayer Perceptron')
    print('2 - Scikit Multilayer Perceptron')
    print('Anyone - Exit')

    option = int(input('Option: '))

    # Perform action
    if option == 1:
        multilayer_perceptron(train_input, train_label, val_input, val_label, test_input, test_label)
    elif option == 2:
        scikit_multilayer_perceptron(train_input, train_label, val_input, val_label, test_input, test_label)


if __name__ == '__main__':
    main()
