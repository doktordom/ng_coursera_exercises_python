import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ex1.linear_regression import read_csv_data


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
    colors = iter(cm.rainbow(np.linspace(0, 1, len(x_data))))
    for x, y, label in zip(x_data, y_data, data_labels):
        plt.scatter(x, y, color=next(colors), label=label)
    plt.plot(fit_x, fit_y)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    legend = plt.legend()  # Create the legend and then set it to be transparent.
    legend.get_frame().set_alpha(0.5)
    plt.show()


def main():
    """
    Runs the main code in this file.
    """
    data1 = read_csv_data('ex2data1.txt')
    # Now need to separate the data into the two different sets.
    x_data = [[data[0] for data in zip(data1[0], data1[2]) if int(data[1]) == 0]]
    y_data = [[data[0] for data in zip(data1[1], data1[2]) if int(data[1]) == 0]]
    x_data.append([data[0] for data in zip(data1[0], data1[2]) if int(data[1]) == 1])
    y_data.append([data[0] for data in zip(data1[1], data1[2]) if int(data[1]) == 1])
    data_labels = ['Not admitted', 'Admitted']
    plot_data_scatter(x_data, y_data, data_labels, x_axis_label='Exam 1 score', y_axis_label='Exam 2 score')

if __name__ == "__main__":
    main()