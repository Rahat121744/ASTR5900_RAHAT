import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

data = np.loadtxt('data.txt', dtype=int)
theta = np.linspace(0, 1, 2000)

plt.figure(figsize=(8,5))

for N in [5, 50, 500]:
    subset = data[:N]
    h = int(np.sum(subset))
    t = N - h
    a = 1 + h
    b = 1 + t

    plt.plot(theta, beta.pdf(theta, a, b),
             label=f'N={N}, Beta({a},{b})')

plt.axvline(0.5, linestyle='--', label='Fair coin')
plt.title('Posterior Comparison')
plt.xlabel('theta')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.show()