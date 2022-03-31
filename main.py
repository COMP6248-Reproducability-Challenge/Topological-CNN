#!/usr/bin/env python
import klein
import numpy as np

if __name__ == "__main__":
    # There are probably some combinations of thetas that give identical filters
    # so we can reduce this list
    #thetas = [0, np.pi, -np.pi, np.pi/2, -np.pi/2, np.pi/4, -np.pi/4]
    thetas = [(i*np.pi)/8 for i in range(16)]
    thetas = [(i*np.pi)/2 for i in range(4)]
    klein.display_kernels(thetas, 5, circle=False)
