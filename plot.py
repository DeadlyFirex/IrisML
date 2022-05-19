# Import libraries
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

# Define class names and import data from file
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = np.genfromtxt('input/iris.csv', delimiter=',', dtype=None, encoding="utf-8", names=names)

# Define X and Y values
x = data['class']
y = data['sepal_length']

# xSmooth = np.linspace(xPoints.min(), yPoints.max())
# spl = make_interp_spline(xPoints, yPoints, k=3)
# ySmooth = spl(xSmooth)
# plt.plot(xSmooth, ySmooth, label='smooth')

# Plot the result in a gridded graph
plt.plot(x, y, label="linear")
plt.grid()
plt.legend(loc='best')
plt.interactive(True)
plt.show()
