import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Known data
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([-1, 1, 2, 3, 1, 4])

# Create interpolation function
f = interp1d(x, y, kind='linear')

# New x values
x_new = np.linspace(0, 5, 30)

# Interpolated y values
y_new = f(x_new)

# Plot
plt.plot(x, y, 'o', label='Original data')
plt.plot(x_new, y_new, '-', label='Interpolated')
plt.title('Interpolation function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('Interpolation function_plot.png')