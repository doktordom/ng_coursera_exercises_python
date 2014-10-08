import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ex1.linear_regression import read_csv_data


def line_plot(x, y, x_label, y_label):
    """
    :param x: list of the horizontal coordinates of the data.
    :param y:  A list of the vertical coordinates of the data.
    :param x_label: Label for the horizontal axis.
    :param y_label: Label for the vertical axis.
    """
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_data_scatter(x_data, y_data, data_labels, x_axis_label, y_axis_label, fit_x=[], fit_y=[]):
    """
    Plots data on a 2D scatter plot.
    :param x_data: A list of lists of the horizontal coordinates of the data.
    :param y_data: A list of lists of the vertical coordinates of the data.
    :param x_axis_label:
    :param y_axis_label:
    :param x_fit:
    :param y_fit:
    """
    # In order to accept an arbitrary number of data sets, make an iterator over a set of unique colours.
    colors = iter(cm.rainbow(np.linspace(0, 1, len(x_data))))
    for x, y, label in zip(x_data, y_data, data_labels):
        plt.scatter(x, y, color=next(colors), label=label)
    plt.plot(fit_x, fit_y)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    legend = plt.legend()  # Create the legend and then set it to be transparent.
    legend.get_frame().set_alpha(0.5)
    plt.show()


def sigmoid(theta, x):
    """
    Apply the sigmoid function to input vectors.
    :param theta: numpy array containing the parameters.
    :param x: numpy array containing the input.
    :return result: The probability returned by applying the sigmoid function to theta and x
    """
    result = 1 / (1 + np.exp(-1 * np.dot(theta.transpose(), x)))
    return result


def logistic_regression(x_data, y_data, alpha, num_iterations):
    """
    Runs logistic regression on a set of input data and corresponding class labels via gradient descent.
    :param x_data: A matrix containing the input data.
    :param y_data: A vector containing the labels, restricted to either 0 or 1 exactly.
    :returns theta: The optimal parameters.
    """
    theta = np.zeros(x_data.shape[0])
    m = y_data.shape[0]
    print 'm', m
    cost_per_epoch = []
    for epoch in range(num_iterations):
        # Calculate the hypothesis.
        hypothesis = sigmoid(theta, x_data)
        # Calculate the cost.
        cost = (1./m) * np.sum((-y_data) * np.log(hypothesis) - (1-y_data) * np.log(1-hypothesis))
        cost_per_epoch.append(cost)
        print cost
        # Do the gradient update.
        theta = theta - alpha * 1./m * np.sum((hypothesis-y_data) * x_data, 1)
    return theta, cost_per_epoch


def admission_regression():
    """
    Runs logistic regression on the exam score data ex2data1.txt
    """
    data1 = read_csv_data('ex2data1.txt')
    # Now need to separate the data into the two different sets.
    x_data = [[data[0] for data in zip(data1[0], data1[2]) if int(data[1]) == 0]]
    y_data = [[data[0] for data in zip(data1[1], data1[2]) if int(data[1]) == 0]]
    x_data.append([data[0] for data in zip(data1[0], data1[2]) if int(data[1]) == 1])
    y_data.append([data[0] for data in zip(data1[1], data1[2]) if int(data[1]) == 1])
    data_labels = ['Not admitted', 'Admitted']
    # Plot the data.
    plot_data_scatter(x_data, y_data, data_labels, x_axis_label='Exam 1 score', y_axis_label='Exam 2 score')

    # Add bias array to original data.
    data1.insert(0, np.ones(data1[0].shape))
    x_data_np = np.vstack(tuple(data1[:-1]))
    y_data_np = data1[-1]

    # Normalize the data.
    normalization_constants = []
    for column in x_data_np[1:]:  # Don't normalize the bias term.
        mean = np.mean(column)
        column -= mean  # Subtract mean.
        standard_deviation = np.std(column)
        if standard_deviation != 0.0:  # Avoid dividing by zero.
            column /= standard_deviation  # Divide by standard deviation.
        normalization_constants.append({'mean': mean, 'std': standard_deviation})
    # Don't normalize the y_data because we know it is 0 or 1

    # Set hyper parameters.
    alpha = 0.1  # Learning rate.
    num_iterations = 2000  # Number of updates before quitting.

    optimal_theta, cost_per_epoch = logistic_regression(x_data_np, y_data_np, alpha, num_iterations)
    # Plot the cost per epoch.
    line_plot(range(len(cost_per_epoch)), cost_per_epoch, 'Epoch', 'Cost')
    
    # Plot the original data with the decision boundary.
    # Need to un-normalize theta, to get it back into the original form of the data.
    orig_theta = [0] * 3
    orig_theta[0] = optimal_theta[0] - \
                    (optimal_theta[1]*normalization_constants[0]['mean']) / normalization_constants[0]['std'] - \
                    (optimal_theta[2]*normalization_constants[1]['mean']) / normalization_constants[1]['std']
    orig_theta[1] = optimal_theta[1] / normalization_constants[0]['std']
    orig_theta[2] = optimal_theta[2] / normalization_constants[1]['std']
    # Generate a fit to the data. Derived from y = h_theta(x) setting y = 0.5.
    x1 = np.asarray(range(int(min(x_data[0])), int(max(x_data[0]))))
    x2 = -(orig_theta[1] / orig_theta[2]) * x1 - (orig_theta[0] / orig_theta[2])
    # x2 = -(optimal_theta_orig[1] / optimal_theta_orig[2]) * x1 + 130
    plot_data_scatter(x_data, y_data, data_labels, x_axis_label='Exam 1 score', y_axis_label='Exam 2 score',
                      fit_x=x1, fit_y=x2)


def main():
    admission_regression()


if __name__ == "__main__":
    main()