import numpy as np
import matplotlib.pyplot as plt
from ex1.linear_regression import read_csv_data


def plot_data_scatter(x_data, y_data, x_axis_label, y_axis_label, fit_x=[], fit_y=[]):
    """
    Plots data on a 2D scatter plot.
    :param x_data: A list of lists of the horizontal coordinates of the data.
    :param y_data: A list of lists of the vertical coordinates of the data.
    :param x_axis_label:
    :param y_axis_label:
    :param x_fit:
    :param y_fit:
    """
    for x, y in zip(x_data, y_data):
        plt.scatter(x, y)
    plt.plot(fit_x, fit_y)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.show()


def main():
    """
    Runs the main code in this file.
    """
    data1 = read_csv_data('ex2data1.txt')
    plot_data_scatter(data1[0], data1[1], 'Exam 1 score', 'Exam 2 score')

if __name__ == "__main__":
    main()