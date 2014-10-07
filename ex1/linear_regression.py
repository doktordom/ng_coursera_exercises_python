import numpy as np
import matplotlib.pyplot as plt


def plot_data_scatter(x, y, x_label, y_label, x_fit=[], y_fit=[]):
    """
    Plots data on a 2D scatter plot.
    :param x: A list of the horizontal coordinates of the data.
    :param y: A list of the vertical coordinates of the data.
    :param x_label: Label for the horizontal axis.
    :param y_label: Label for the vertical axis.
    :param x_fit: List of horizontal coordinates for a trend line.
    :param y_fit: List of vertical coordinates for a trend line.
    """
    plt.scatter(x, y)
    plt.plot(x_fit, y_fit)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def linear_regression(x_data, y_data, alpha, num_iterations):
    """
    Performs batch gradient descent on the input data using the hypothesis h = theta_0 + theta_1 * x_1
    :param x_data: A matrix containing the input data.
    :param y_data: A vector containing the label.
    :returns theta: The optimal parameters.
    """
    theta = np.zeros(x_data.shape[0])
    m = y_data.shape[0]
    cost_per_epoch = []
    for epoch in range(num_iterations):
        error = np.dot(theta.transpose(), x_data) - y_data
        cost = 1./(2*m) * np.sum(error ** 2)
        cost_per_epoch.append(cost)
        theta = theta - alpha * 1./m * np.sum(error * x_data, 1)
        print cost
    return theta, cost_per_epoch


def read_csv_data(file_path):
    """
    Reads in data with an arbitrary number of comma separated columns.
    :param file_path: The path to the data file to be read in.
    :return data: The data as a list of numpy arrays. Each array is a column of data.
    """
    with open(file_path, 'r') as input_data:
        # split_data contains all the rows from the data file.
        split_data = [row.split(',') for row in input_data]
    # create a list of values for each column in the data file, put them in data.
    data = []
    for i in range(len(split_data[0])):
        data.append(np.asarray([float(row[i]) for row in split_data]))
    return data


def food_truck():
    """
    Runs linear regression on the food truck data.
    """
    # Read the data from ex1data1.txt file.
    with open('ex1data1.txt', 'r') as input_data:
        split_data = [(float(row.split(',')[0]), float(row.split(',')[1])) for row in input_data]
        populations = [i[0] for i in split_data]
        profits = [i[1] for i in split_data]
    # Plot the data.
    # plot_data_scatter(populations, profits, 'Population', 'Profit')

    # Create the data matrices including the bias term.
    x_data = np.vstack((np.ones(len(populations)), np.asarray(populations)))
    y_data = np.asarray(profits)

    # Set hyper parameters.
    alpha = 0.01  # Learning rate.
    num_iterations = 1500  # Number of updates before quitting.
    optimal_theta, cost_per_epoch = linear_regression(x_data, y_data, alpha, num_iterations)

    # Plot data with a line fitting based on the optimal theta.
    x_fit = [min(populations), max(populations)]
    y_fit = [optimal_theta[0] + optimal_theta[1]*x for x in x_fit]
    plot_data_scatter(populations, profits, 'Population', 'Profit', x_fit, y_fit)


def house_portland():
    """
    Runs linear regression on multiple variables on the Portland house data.
    """
    data = read_csv_data('ex1data2.txt')
    # Add bias array.
    data.insert(0, np.ones(data[0].shape))
    x_data = np.vstack(tuple(data[:-1]))
    y_data = data[-1]
    # Normalize the data.
    normalization_constants = []
    for column in x_data:
        mean = np.mean(column)
        column -= mean  # Subtract mean.
        standard_deviation = np.std(column)
        if standard_deviation != 0.0:  # Avoid dividing by zero.
            column /= standard_deviation  # Divide by standard deviation.
        normalization_constants.append({'mean': mean, 'std': standard_deviation})
    # Also normalize y_data.
    y_mean = np.mean(y_data)
    y_data -= y_mean
    y_std = np.std(y_data)
    y_data /= y_std
    y_norms = {'mean': y_mean, 'std': y_std}

    # Set hyper parameters.
    alpha = 0.01  # Learning rate.
    num_iterations = 1500  # Number of updates before quitting.
    optimal_theta, cost_per_epoch = linear_regression(x_data, y_data, alpha, num_iterations)

    plot_data_scatter(range(len(cost_per_epoch)), cost_per_epoch, 'epoch', 'cost')

    # Apply optimal theta to a new input.
    new_input = [1, 1650, 3]
    # Normalize it.
    for index, value, norm in zip(range(len(new_input)), new_input, normalization_constants):
        value -= norm['mean']
        if norm['std'] != 0.0:
            value /= norm['std']
        new_input[index] = value
    norm_prediction = np.dot(optimal_theta, np.asarray(new_input))
    prediction = norm_prediction * y_norms['std']
    prediction += y_norms['mean']

    print "Price prediction for a house of 1650 square feet and 3 bedrooms is:", prediction


def house_normal_equations():
    """
    Calculates the optimal value for theta in the house data set using the normal equation.
    """
    data = read_csv_data('ex1data2.txt')
    # Add bias array.
    data.insert(0, np.ones(data[0].shape))
    x_data = np.vstack(tuple(data[:-1]))
    x_data = x_data.transpose()
    y_data = data[-1]
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_data.transpose(), x_data)), x_data.transpose()), y_data)
    print theta
    new_input = [1, 1650, 3]
    prediction = np.dot(theta, np.asarray(new_input))
    print prediction

if __name__ == "__main__":
    food_truck()
    house_portland()
    house_normal_equations()