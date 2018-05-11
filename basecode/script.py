import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.svm import SVC



def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    # error = 1
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################

    w = initialWeights
    w = w.reshape((n_feature+1, 1))

    bias = np.ones((n_data, 1))
    x = np.append(bias, train_data, axis=1)

    theta = sigmoid(x.dot(w))

    error = labeli * np.log(theta) + (1.0 - labeli) * np.log(1.0 - theta)
    # error = error/n_data
    error = - np.sum(error)

    error_grad = (theta - labeli)*x
    error_grad = np.sum(error_grad, axis=0)
    # print(error_grad.shape)
    # HINT: Do not forget to add the bias term to your input data
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    # print(data.shape)
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias = np.ones((data.shape[0], 1))
    x = np.append(bias, data, axis=1)

    label = sigmoid(x.dot(W))  # compute probabilities
    label = np.argmax(label, axis=1)  # get maximum for each class
    label = label.reshape((data.shape[0], 1))


    return label


def mlrObjFunction(initialWeights, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    error_grad.flatten()

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    w = initialWeights
    w = w.reshape((n_feature+1, n_class))

    bias = np.ones((n_data, 1))
    x = np.append(bias, train_data, axis=1)
    # print(x.shape)
    denominator = np.sum(np.exp(x.dot(w)), axis=1)
    denominator = denominator.reshape(n_data, 1)

    posterior = (np.exp(x.dot(w)) / denominator)

    log_theta = np.log(posterior)

    error = (-1) * np.sum(np.sum(labeli * log_theta))
    error_grad = (np.dot(x.T, posterior - labeli))
    # print(error_grad.shape)
    return error, error_grad.flatten()


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    n_data = data.shape[0]
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias = np.ones((n_data, 1))
    x = np.append(bias, data, axis=1)

    denominator = np.sum(np.exp(x.dot(W)))

    posterior = (np.exp(x.dot(W)) / denominator)

    for i in range(posterior.shape[0]):
        label[i] = np.argmax(posterior[i])
    label = label.reshape(label.shape[0], 1)

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    print(i)
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

print("Done training")

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
# cm = confusion_matrix(train_label, predicted_label)
# print(cm)
# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
# cm = confusion_matrix(validation_label, predicted_label)
# print(cm)
# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
# cm = confusion_matrix(test_label, predicted_label)
# print(cm)
"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

# Selecting random 10000 points
random_points = np.random.choice(50000, 10000)
train_data = train_data[random_points]
train_label = train_label[random_points]

# Selecting random 2000 points
random_points = np.random.choice(10000, 2000)
validation_data = validation_data[random_points]
validation_label = validation_label[random_points]
test_data = test_data[random_points]
test_label = test_label[random_points]

# Linear Kernel
print('#----------------------Using linear kernel----------------------#')
clf = SVC(kernel='linear')
print('\n Fitting...')
clf.fit(train_data, train_label.flatten())
print('\n Fitting for Linear Kernel Completed')
print('\n Training set Accuracy:', clf.score(train_data, train_label))
print('\n Validation set Accuracy:', clf.score(validation_data, validation_label))
print('\n Testing set Accuracy:', clf.score(test_data, test_label))

# Radial basis function with gamma = 0

print('\n\n #----------------------Using radial basis function with default gamma----------------------#')
clf = SVC(kernel='rbf')
print('\n Fitting...')
clf.fit(train_data, train_label.flatten())
print('\n Fitting for rbf with default gamma Completed')
print('\n Training set Accuracy:', clf.score(train_data, train_label))
print('\n Validation set Accuracy:', clf.score(validation_data, validation_label))
print('\n Testing set Accuracy:', clf.score(test_data, test_label))

# Radial basis function with gamma = 1

print('\n\n #----------------------Using radial basis function with gamma = 1----------------------#')
clf = SVC(kernel='rbf', gamma=1.0)
print('\n Fitting...')
clf.fit(train_data, train_label.flatten())
print('\n Fitting for rbf with gamma = 1 Completed')
print('\n Training set Accuracy:', clf.score(train_data, train_label))
print('\n Validation set Accuracy:', clf.score(validation_data, validation_label))
print('\n Testing set Accuracy:', clf.score(test_data, test_label))

# Radial basis function with C = 1, 10, 20 ... 100

print('\n\n #----------------------Using radial basis function with different values of C----------------------#')

c_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
train_accuracy = np.zeros(len(c_values))
valid_accuracy = np.zeros(len(c_values))
test_accuracy = np.zeros(len(c_values))


for i in range(len(c_values)):
    print("\n Fitting for", c_values[i])
    clf = SVC(C=c_values[i],kernel='rbf')
    clf.fit(train_data, train_label.flatten())
    train_accuracy[i] = 100*clf.score(train_data, train_label)
    print("\n When C =", c_values[i], "Training set Accuracy: ", train_accuracy[i])
    valid_accuracy[i] = 100*clf.score(validation_data, validation_label)
    print("\n When C =", c_values[i], "Validation set Accuracy: ", valid_accuracy[i])
    test_accuracy[i] = 100*clf.score(test_data, test_label)
    print("\n When C =", c_values[i], "Test set Accuracy: ", test_accuracy[i])

accuracyMatrix = np.column_stack((train_accuracy, valid_accuracy, test_accuracy))

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(c_values, accuracyMatrix)
plt.title('Accuracy with different C')
plt.legend(('Training data', 'Validation data', 'Testing data'), loc='best')
plt.xlabel('Values of C')
plt.ylabel('Accuracy in %')
plt.show()

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
# cm = confusion_matrix(train_label, predicted_label_b)
# print(cm)

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
# cm = confusion_matrix(validation_label, predicted_label_b)
# print(cm)

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
# cm = confusion_matrix(test_label, predicted_label_b)
# print(cm)