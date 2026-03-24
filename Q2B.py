import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Load data
data = np.loadtxt('data.txt', dtype=int)

# Counts
n = len(data)
h = int(np.sum(data))
t = n - h

print('Total flips =', n)
print('Heads =', h)
print('Tails =', t)

# Posterior
theta = np.linspace(0, 1, 2000)
a_post = 1 + h
b_post = 1 + t

plt.figure(figsize=(8,5))
plt.plot(theta, beta.pdf(theta, a_post, b_post), label=f'Beta({a_post},{b_post})')
plt.axvline(0.5, linestyle='--', label='Fair coin')
plt.axvline(a_post/(a_post+b_post), linestyle='dotted', label='Mean')
plt.title('Posterior Distribution (500 flips)')
plt.xlabel('theta')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.show()