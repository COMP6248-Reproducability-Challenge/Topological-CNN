import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import os

def Q(t: float) - float:
    return 2*t**2 - 1

# Calculates Klein Filter value for position (x,y)
def klein_filter(theta1: float, theta2: float, x: int, y: int) -> float:
    return np.sin(theta2)*(np.cos(theta1)*x+ np.sin(theta1)*y) + np.cos(theta2)*Q(np.cos(theta1)*x + np.sin(theta1)*y)

# Generates a Circle Filter value for position (x,y)
def primary_circle(theta: float, x: int, y: int) -> float:
    return klein_filter(theta, np.pi /2, x,y)

# Generates a complete Klein or Primary Circle filter/kernel
def generate_kernel(size: int, theta1: float, theta2 = np.pi /2) -> np.ndarray:
    results = []
    # This linspace seems to work for KFs but not CFs
    # We might need to find a different way of choosing our x,y values
    linspace = np.linspace(-1,1,size)
    for x in linspace :
        row = []
        for y in linspace :
            kf = klein_filter(theta1, theta2, x, y)
            row.append(kf)
        results.append(row)
    results = np.array(results)
    return results 

# Displays all combinations
def display_kernels(thetas: [float], size: int) -> None:

    try:
        os.mkdir('fig')
    except:
        pass

    for theta1 in thetas:
        for theta2 in thetas:
            kfilter = generate_kernel(size, theta1, theta2)
            ax = sns.heatmap(kfilter, cmap=cm.gray)
            ax.set_title("Theta1: {}\u03c0 and Theta2: {}\u03c0".format(theta1 / np.pi, theta2 / np.pi))
            plt.savefig('fig/th1-{}\u03c0_th2-{}\u03c0.pdf'.format(theta1 / np.pi, theta2/ np.pi))
            plt.show()
