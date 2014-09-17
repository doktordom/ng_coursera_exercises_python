import numpy as np
import matplotlib.pyplot as plt


def plot_data_scatter(x, y, x_label, y_label):
    """
    Plots data on a 2D scatter plot.
    :param x: A list of the horizontal coordinates of the data.
    :param y: A list of the vertical coordinates of the data.
    """
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def linear_regression(x_data, y_data, alpha, num_iterations):
    """
    Performs batch gradient descent on the input data using the hypothesis h = theta_0 + theta_1 * x_1
    :param x_data: A list of the horizontal coordinates of the data.
    :param y_data: A list of the vertical coordinates of the data.
    :returns theta: The optimal parameters.
    """
    theta = np.zeros(2)
    m = y_data.shape[0]
    for epoch in range(num_iterations):
        error = np.dot(theta.transpose(), x_data) - y_data
        cost = 1./(2*m) * np.sum(error ** 2)
        theta = theta - alpha * 1./m * np.sum(error * x_data, 1)
        print cost

    return theta


def main():
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

    # Set hyper parameters
    alpha = 0.01  # Learning rate.
    num_iterations = 1500  # Number of updates before quitting.
    optimal_theta = linear_regression(x_data, y_data, alpha, num_iterations)

    # Plot data with a line fitting based on the optimal theta


if __name__ == "__main__":
    main()