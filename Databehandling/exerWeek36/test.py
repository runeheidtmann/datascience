import matplotlib.pyplot as plt
import numpy as np

def drawMap():
  #load csv file with delimiter ';'
  data = np.genfromtxt("earthquakes.csv", delimiter=';')
  #slicing: filtering rows with at least intensity 8
  data = data[ data[:,2] >= 8 ]
  #longitudes for x and latitudes for y
  x = data[:,5]
  y = data[:,4]
  plt.scatter(x,y)
  plt.show()

drawMap()