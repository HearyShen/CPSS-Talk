"""
Demo of Runge Phenomenon

Author: HearyShen
Date: 2020.10.28
"""
import matplotlib.pyplot as plt
import numpy as np


LEFT_BOUND = -1
RIGHT_BOUND = 1.01

PLT_ROWS = 2
PLT_COLS = 3


def runge_function(x):
    y = 1 / (1 + 25 * x**2)     # Runge Function
    return y


def runge_polyfit(degree):
    # original runge function
    x_orig = np.arange(LEFT_BOUND, RIGHT_BOUND, 0.01)
    y_orig = runge_function(x_orig)

    # sample points
    interval = 2 / degree
    x_sample = np.arange(LEFT_BOUND, RIGHT_BOUND, interval)
    y_sample = runge_function(x_sample)

    plt.title(f"{degree}-degree polyfit")

    # plot original runge function
    plt.plot(x_orig, y_orig, label="runge")

    # plot runge sample points
    plt.plot(x_sample, y_sample, "rx")

    coef = np.polyfit(x_sample, y_sample, degree)
    # y_fit = sum([coef[i] * (x**(degree-i)) for i in range(degree+1)])
    y_fit = np.polyval(coef, x_orig)
    plt.plot(x_orig, y_fit, label=f"polyfit-{degree}")

    plt.legend()
    # plt.show()


def runge_phenomenon(degrees=[1, 5, 9, 13, 15, 17]):
    plt.figure(figsize=(16, 9))
    for i in range(min(PLT_ROWS*PLT_COLS, len(degrees))):
        # plot each polynomial function of each degree
        plt.subplot(PLT_ROWS, PLT_COLS, i+1)

        # plot runge function
        runge_polyfit(degrees[i])

    plt.show()


if __name__ == "__main__":
    runge_phenomenon()
