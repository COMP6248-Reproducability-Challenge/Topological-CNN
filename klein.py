import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import os
from scipy.integrate import dblquad

def Q(t: float) -> float:
    return (2*t)**2 - 1

# Calculates Klein Filter value for position (x,y)
def klein(y: int, x: int, theta1: float, theta2: float) -> float:
    return np.sin(theta2)*(np.cos(theta1)*x+ np.sin(theta1)*y) + np.cos(theta2)*Q(np.cos(theta1)*x + np.sin(theta1)*y)

# Generates a Circle Filter value for position (x,y)
def primary_circle(y: int, x: int,theta: float) -> float:
    return klein(theta, np.pi/2, x, y)

#generates klein of specific size and angle
def generate_klein_filter(size: int, theta1: float, theta2: float) -> np.ndarray:

    klein_filter = np.zeros([size,size])
    for x in range(size):
        for y in range(size):
           xmin = -1.0 + ((2*x) / (2*size))
           xmax = -1.0 + ((2*(x+1.0)) / (2*size))
           ymin = -1.0 + ((2*y) / (2*size))
           ymax = -1.0 + ((2*(y+1.0)) /(2*size))


           Filter = dblquad(klein,xmin,xmax,ymin,ymax, args=(theta1,theta2))[0]
           klein_filter[y][x] = Filter
    return np.array(klein_filter)

#generates primary circle filter with specific size and angle
def generate_pc_filter(size: int, theta1: float) -> np.ndarray:

    pc_filter = np.zeros([size,size])
    for x in range(size):
        for y in range(size):
           xmin = -1.0 + ((2*x) / (2*size))
           xmax = -1.0 + ((2*(x+1.0)) / (2*size))
           ymin = -1.0 + ((2*y) / (2*size))
           ymax = -1.0 + ((2*(y+1.0)) /(2*size))


           Filter = dblquad(primary_circle,xmin,xmax,ymin,ymax, args=((theta1,)))[0]
           pc_filter[y][x] = Filter
    return np.array(pc_filter)


# Displays all combinations
def display_kernels(thetas: [float], size: int, thetas2 = None, circle = False) -> None:

    try:
        os.mkdir('fig')
    except:
        pass

    if not circle:
        for theta1 in thetas:
            for theta2 in thetas2:
                kfilter = generate_klein_filter(size, theta1, theta2)
                ax = sns.heatmap(kfilter, cmap=cm.gray)
                ax.set_title("Theta1: {}\u03c0 and Theta2: {}\u03c0".format(theta1 / np.pi, theta2 / np.pi))
                plt.savefig('fig/th1-{}\u03c0_th2-{}\u03c0.pdf'.format(theta1 / np.pi, theta2/ np.pi))
                plt.show()
    else:
        for theta1 in thetas:
                kfilter = generate_pc_filter(size, theta1)
                ax = sns.heatmap(kfilter, cmap=cm.gray)
                ax.set_title("Theta1: {}\u03c0 and Theta2: {}\u03c0".format(theta1 / np.pi, 0.5))
                plt.savefig('fig/th1-{}\u03c0_th2-{}\u03c0.pdf'.format(theta1 / np.pi, 0,5))
                plt.show()
