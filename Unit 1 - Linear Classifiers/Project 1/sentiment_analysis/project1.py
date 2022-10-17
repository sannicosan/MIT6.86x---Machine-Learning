from string import punctuation, digits
import numpy as np
import random

# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """

    ## Agreements function
    y = label
    x = feature_vector
    z = y*(np.dot(theta,x) + theta_0)

    ## Loss function ~ Hinge loss
    loss_h = 0 if z>=1 else 1-z

    return loss_h
    raise NotImplementedError


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """

    ## Agreements function
    y = labels
    x = feature_matrix
    z = y * (np.dot(theta, x.T) + theta_0)

    ## Loss function ~ Hinge loss
    loss = lambda l: 0 if l>=1 else 1-l
    loss_function = np.array([loss(zi) for zi in z])
    loss_h = np.average(loss_function)
    return loss_h

    raise NotImplementedError


def perceptron_single_step_update(feature_vector,label,current_theta,current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """

    x = feature_vector
    y = label
    theta = current_theta
    theta_0 = current_theta_0

    z = y * (np.dot(theta, x.T) + theta_0)

    # theta = theta + y*x
    # theta_0 = theta_0 + y

    if (np.abs(z) <= 1e-16 or z < 0):
        theta = theta + y*x
        theta_0 = theta_0 + y

    return (theta,theta_0)


    raise NotImplementedError


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    ## Initialization
    n = len(feature_matrix[0])
    theta = np.zeros_like(feature_matrix[0])
    theta_0 = 0
    x = feature_matrix
    y = labels
    
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            (theta,theta_0) = perceptron_single_step_update(feature_matrix[i],labels[i], theta,theta_0)
            pass
    return (theta,theta_0)
    raise NotImplementedError


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    ## Initialization
    m = feature_matrix.shape[0]
    d = feature_matrix.shape[1]
    theta = np.zeros_like(feature_matrix[0])
    theta_0 = 0
    theta_tup = (theta,theta_0)
    theta_sum = theta
    theta_0_sum = theta_0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta_tup = perceptron_single_step_update(feature_matrix[i], labels[i], theta_tup[0], theta_tup[1])
            theta_sum = theta_sum + theta_tup[0]
            theta_0_sum += theta_tup[1]
            pass
    return (theta_sum/(m*T), theta_0_sum/(m*T))
    raise NotImplementedError



def pegasos_single_step_update(feature_vector,label,L,eta,current_theta,current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    x = feature_vector
    y = label
    theta = current_theta
    theta_0 = current_theta_0

    z = y * (np.dot(theta, x.T) + theta_0)

    # theta = theta + y*x
    # theta_0 = theta_0 + y

    if (z <= 1):
        theta = (1-eta*L)*theta + eta*y * x
        theta_0 = theta_0 + eta*y
    else:
        theta =  (1 - eta * L)* theta
        theta_0 = theta_0
    return (theta, theta_0)

    raise NotImplementedError


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    ## Initialization
    m = feature_matrix.shape[0]
    d = feature_matrix.shape[1]
    x = feature_matrix
    y = labels
    theta = np.zeros_like(feature_matrix[0])
    theta_0 = 0
    theta_tup = (theta, theta_0)
    t = 0

    ## Full Perceptron
    for tx in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # z = y[i] * (np.dot(theta_tup[0], x[i].T) + theta_tup[1])

            t += 1
            eta = 1 / np.sqrt(t)
            theta_tup = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, theta_tup[0], theta_tup[1])

    return (theta_tup[0],theta_tup[1])
    raise NotImplementedError

# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """

    y = np.zeros(feature_matrix.shape[0])
    for i,xi in enumerate(feature_matrix):
        pred = np.dot(theta,xi) + theta_0
        y[i] = 1 if pred >0 else -1

    return y

    raise NotImplementedError


def classifier_accuracy(classifier,train_feature_matrix,val_feature_matrix,train_labels,val_labels,**kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """

    model_params = classifier(train_feature_matrix,train_labels,**kwargs) # model_params = (theta,theta_0)
    y_preds_train = classify(train_feature_matrix,model_params[0],model_params[1])
    y_preds_val = classify(val_feature_matrix, model_params[0], model_params[1])

    acc_train = accuracy(y_preds_train,train_labels)
    acc_val = accuracy(y_preds_val,val_labels)

    return (acc_train,acc_val)


    raise NotImplementedError


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    ## Load stop_words as a list
    stop_words_list = []
    with open('stopwords.txt') as file:
        for line in file:
            stop_words_list.append(line.split()[0])

    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if (word not in dictionary and word not in stop_words_list):
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = word_list.count(word)
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
