import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

data = np.loadtxt('data.txt', dtype=int)
theta = np.linspace(0, 1, 2000)

priors = [(1,1), (10,10), (8,2), (2,8)]

for N in [5, 50, 500]:
    subset = data[:N]
    h = int(np.sum(subset))
    t = N - h

    plt.figure(figsize=(8,5))

    for alpha, beta_prior in priors:
        a = alpha + h
        b = beta_prior + t

        plt.plot(theta, beta.pdf(theta, a, b),
                 label=f'Beta({alpha},{beta_prior})')

    plt.axvline(0.5, linestyle='--', label='Fair coin')
    plt.title(f'Effect of Prior (N={N})')
    plt.xlabel('theta')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.show()