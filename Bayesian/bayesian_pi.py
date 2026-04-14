import numpy as np
import matplotlib.pyplot as plt
from math import gamma, pi


# Bayesian estimation of pi (Problems 3A-3C)


def beta_function(a, b):
    return gamma(a) * gamma(b) / gamma(a + b)


def beta_pdf(x, a, b):
    B = beta_function(a, b)
    return (x ** (a - 1)) * ((1 - x) ** (b - 1)) / B


def generate_points(N=100, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.random(N)
    y = rng.random(N)
    inside = (x**2 + y**2) <= 1
    return x, y, inside


def problem_3ab():
    x, y, inside = generate_points(N=100, seed=42)
    N = len(x)
    m = int(np.sum(inside))
    outside = N - m

    print('--- Problem 3A ---')
    print(f'Inside points = {m}')
    print(f'Outside points = {outside}')

    xx = np.linspace(0, 1, 500)
    yy = np.sqrt(1 - xx**2)

    plt.figure(figsize=(6, 6))
    plt.scatter(x[inside], y[inside], label='Inside')
    plt.scatter(x[~inside], y[~inside], label='Outside')
    plt.plot(xx, yy, label='Circle')
    plt.title('Monte Carlo for Pi')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('Q3.png', dpi=300)
    plt.close()

    # Problem 3B
    alpha, beta_prior = 1, 1
    a_post = alpha + m
    b_post = beta_prior + (N - m)
    p_mean = a_post / (a_post + b_post)
    pi_est = 4 * p_mean

    print('\n--- Problem 3B ---')
    print(f'Posterior Beta({a_post},{b_post})')
    print(f'Estimated pi = {pi_est:.4f}')

    pgrid = np.linspace(1e-4, 1 - 1e-4, 2000)
    plt.figure(figsize=(8, 5))
    plt.plot(pgrid, beta_pdf(pgrid, a_post, b_post))
    plt.axvline(p_mean, linestyle='dotted', label='Mean')
    plt.axvline(pi / 4, linestyle='--', label='True value')
    plt.title('Posterior of p')
    plt.xlabel('p')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Q3_P_of_P.png', dpi=300)
    plt.close()

    return pi_est


def bayesian_pi_estimate(num_points, alpha, beta_prior, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.random(num_points)
    y = rng.random(num_points)
    m = int(np.sum((x**2 + y**2) <= 1))
    p_mean = (alpha + m) / (alpha + beta_prior + num_points)
    return 4 * p_mean


def problem_3c():
    print('\n--- Problem 3C ---')
    sample_sizes = [10, 30, 100, 300, 1000, 3000]
    priors = [(1, 1), (10, 10), (8, 2)]

    plt.figure(figsize=(8, 5))
    for alpha, beta_prior in priors:
        errors = []
        for N in sample_sizes:
            pi_hat = bayesian_pi_estimate(N, alpha, beta_prior)
            rel_error = abs(pi_hat - pi) / pi
            errors.append(rel_error)
        print(f'Prior Beta({alpha},{beta_prior}) errors = {[round(e,4) for e in errors]}')
        plt.plot(sample_sizes, errors, marker='o', label=f'Prior Beta({alpha},{beta_prior})')

    plt.xscale('log')
    plt.xlabel('Number of random points N')
    plt.ylabel('Relative error')
    plt.title('Relative Error of Bayesian pi Estimate vs Number of Points')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pi_relative_error.png', dpi=300)
    plt.close()


def main():
    problem_3ab()
    problem_3c()


if __name__ == '__main__':
    main()
