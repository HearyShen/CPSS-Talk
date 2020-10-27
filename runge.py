import matplotlib.pyplot as plt
import numpy as np


LEFT_BOUND = -1.5
RIGHT_BOUND = 1.5


def runge_function(x):
    y = 1 / (1 + 25 * x**2)     # Runge Function
    return y

def runge_pheno(degree):
    # original runge function
    x_orig = np.arange(LEFT_BOUND, RIGHT_BOUND, 0.01)
    y_orig = runge_function(x_orig)

    # sample points
    interval = 2 / degree
    x_sample = np.arange(LEFT_BOUND, RIGHT_BOUND, interval)
    y_sample = runge_function(x_sample)

    plt.title(f"Runge {degree}D Ploynomial")
    
    # plot original runge function
    plt.plot(x_orig, y_orig, label="runge")

    # plot runge sample points
    plt.plot(x_sample, y_sample, "rx")

    coef = np.polyfit(x_sample, y_sample, degree)
    # y_fit = sum([coef[i] * (x**(degree-i)) for i in range(degree+1)])
    y_fit = np.polyval(coef, x_orig)
    plt.plot(x_orig, y_fit, label=f"polyfit-{degree}")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # runge_pheno(1)
    # runge_pheno(3)
    runge_pheno(9)