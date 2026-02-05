import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Known data
x = np.array([-1, 0, 1, 2, 3, 3.5, 5, 6, 7, 8])
y = np.array([-3, -1, 1, 3, 3, 1, 4, 2, 1, 1])

# Create interpolation function
f = interp1d(x, y, kind='linear')

# New x values
x_new = np.linspace(-1, 8, 50)

# Interpolated y values
y_new = f(x_new)

# Plot
plt.plot(x, y, 'o', label='Original data')
plt.plot(x_new, y_new, '-', label='Interpolated')
plt.legend()
plt.show()
