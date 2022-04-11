import torch.nn as nn
import torch.nn.functional as F
from klein import generate_klein_filter, generate_pc_filter

class KF_Layer(nn.Module):

   def __init__(self, size: int, slices: int):
       # Need to create sqrt(slices) evenly spaced angles for both thetas
       angle = sqrt(slices)
       thetas1 =  [(i*np.pi)/angle for i in range(angle)]
       thetas2 = [(i*np.pi*2)/angle for i in range(angle)]
       self.filters = []

       for theta2 in thetas2:
           for theta1 in thetas1:
               filters.append(generate_kernel(size, theta1, theta2))


   def forward(self, x):
       output = []
       for filter in self.filters:
           output.append(F.conv2d(x, filter).squeeze(0))
       return torch.tensor(output)


class CF_Layer(nn.Module):

   def __init__(self, size: int, slices: int):
       # Need to generate slices number of evenly spaced angles around a circle
       thetas = [(i*np.pi)/size for i in range(size)]

       self.filters = []

       for theta in thetas:
           filters.append(generate_kernel(size, theta))


   def forward(self, x):
      output = []
      for filter in self.filters:
          output.append(F.conv2d(x, filter).squeeze(0))
      return torch.tensor(output)
