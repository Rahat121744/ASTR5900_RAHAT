import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Generate points
rng = np.random.default_rng(42)
N = 100

x = rng.random(N)
y = rng.random(N)

inside = (x**2 + y**2) <= 1
m = int(np.sum(inside))

print("Inside points =", m)
print("Outside points =", N - m)

# Plot points
xx = np.linspace(0, 1, 500)
yy = np.sqrt(1 - xx**2)

plt.figure(figsize=(6,6))
plt.scatter(x[inside], y[inside], label='Inside')
plt.scatter(x[~inside], y[~inside], label='Outside')
plt.plot(xx, yy, label='Circle')

plt.title("Monte Carlo for Pi")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Bayesian estimation
ap = 1 + m
bp = 1 + (N - m)

p_mean = ap / (ap + bp)
pi_est = 4 * p_mean

print("Posterior Beta(", ap, ",", bp, ")")
print("Estimated pi =", pi_est)

# Posterior plot
pgrid = np.linspace(0, 1, 2000)

plt.figure(figsize=(8,5))
plt.plot(pgrid, beta.pdf(pgrid, ap, bp))
plt.axvline(p_mean, linestyle='dotted', label='Mean')
plt.axvline(np.pi/4, linestyle='--', label='True value')

plt.title("Posterior of p")
plt.xlabel("p")
plt.ylabel("PDF")
plt.legend()
plt.grid(True)
plt.show()

# Relative error vs sample size
def bayesian_pi_estimate(num_points, alpha, beta_prior, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.random(num_points)
    y = rng.random(num_points)
    inside = (x**2 + y**2) <= 1
    m = int(np.sum(inside))

    p_mean = (alpha + m) / (alpha + beta_prior + num_points)
    return 4 * p_mean


sample_sizes = [10, 30, 100, 300, 1000, 3000]
priors = [(1, 1), (10, 10), (8, 2)]

plt.figure(figsize=(8, 5))

for alpha, beta_prior in priors:
    errors = []

    for N in sample_sizes:
        pi_hat = bayesian_pi_estimate(N, alpha, beta_prior)
        rel_error = abs(pi_hat - np.pi) / np.pi
        errors.append(rel_error)

    plt.plot(sample_sizes, errors, marker='o',
             label=f'Prior Beta({alpha},{beta_prior})')

plt.xscale('log')
plt.xlabel('Number of random points N')
plt.ylabel('Relative error')
plt.title('Relative Error of Bayesian pi Estimate vs Number of Points')
plt.legend()
plt.grid(True)
plt.savefig("pi_relative_error.png", dpi=300)
plt.show()