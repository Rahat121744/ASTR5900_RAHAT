import numpy as np
import matplotlib.pyplot as plt
from math import lgamma, exp


# Bayesian inference for coin toss data (Problems 2A-2D)


def beta_function(a, b):
    """Beta function B(a,b) computed safely in log-space."""
    return exp(lgamma(a) + lgamma(b) - lgamma(a + b))


def beta_pdf(x, a, b):
    """Beta(a,b) probability density function computed safely in log-space."""
    log_pdf = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - (lgamma(a) + lgamma(b) - lgamma(a + b))
    return np.exp(log_pdf)


def load_coin_data(filename='data.txt'):
    data = np.loadtxt(filename, dtype=int)
    return data


def count_heads_tails(data):
    h = int(np.sum(data))
    t = len(data) - h
    return h, t


def posterior_parameters(alpha, beta_prior, h, t):
    return alpha + h, beta_prior + t


def posterior_mean(a_post, b_post):
    return a_post / (a_post + b_post)


def plot_posterior(ax, theta, a_post, b_post, label):
    ax.plot(theta, beta_pdf(theta, a_post, b_post), label=label)


def problem_2b(data, alpha=1, beta_prior=1):
    n = len(data)
    h, t = count_heads_tails(data)
    a_post, b_post = posterior_parameters(alpha, beta_prior, h, t)
    mean_theta = posterior_mean(a_post, b_post)

    print('--- Problem 2B ---')
    print(f'Total flips = {n}')
    print(f'Heads = {h}')
    print(f'Tails = {t}')
    print(f'Posterior = Beta({a_post},{b_post})')
    print(f'Posterior mean = {mean_theta:.4f}')
    if abs(mean_theta - 0.5) < 0.02:
        print('Conclusion: The coin is approximately balanced.')
    else:
        print('Conclusion: The coin is biased toward heads.' if mean_theta > 0.5 else 'Conclusion: The coin is biased toward tails.')

    theta = np.linspace(1e-4, 1 - 1e-4, 2000)
    plt.figure(figsize=(8, 5))
    plt.plot(theta, beta_pdf(theta, a_post, b_post), label=f'Beta({a_post},{b_post})')
    plt.axvline(0.5, linestyle='--', label='Fair coin')
    plt.axvline(mean_theta, linestyle='dotted', label='Mean')
    plt.title('Posterior Distribution (500 flips)')
    plt.xlabel('theta')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Q2B.png', dpi=300)
    plt.close()


def problem_2c(data, alpha=1, beta_prior=1):
    print('\n--- Problem 2C ---')
    theta = np.linspace(1e-4, 1 - 1e-4, 2000)

    plt.figure(figsize=(8, 5))
    for N in [5, 50, 500]:
        subset = data[:N]
        h, t = count_heads_tails(subset)
        a_post, b_post = posterior_parameters(alpha, beta_prior, h, t)
        mean_theta = posterior_mean(a_post, b_post)
        print(f'N={N}: Beta({a_post},{b_post}), mean = {mean_theta:.4f}')
        plt.plot(theta, beta_pdf(theta, a_post, b_post), label=f'N={N}, Beta({a_post},{b_post})')

    plt.axvline(0.5, linestyle='--', label='Fair coin')
    plt.title('Posterior Comparison')
    plt.xlabel('theta')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Q2C.png', dpi=300)
    plt.close()


def problem_2d(data, priors=None):
    if priors is None:
        priors = [(1, 1), (10, 10), (8, 2), (2, 8)]

    print('\n--- Problem 2D ---')
    theta = np.linspace(1e-4, 1 - 1e-4, 2000)

    for N in [5, 50, 500]:
        subset = data[:N]
        h, t = count_heads_tails(subset)
        print(f'\nN={N}')
        plt.figure(figsize=(8, 5))

        for alpha, beta_prior in priors:
            a_post, b_post = posterior_parameters(alpha, beta_prior, h, t)
            plt.plot(theta, beta_pdf(theta, a_post, b_post), label=f'Beta({alpha},{beta_prior})')
            print(f'Prior Beta({alpha},{beta_prior}) -> Posterior Beta({a_post},{b_post})')

        plt.axvline(0.5, linestyle='--', label='Fair coin')
        plt.title(f'Effect of Prior (N={N})')
        plt.xlabel('theta')
        plt.ylabel('PDF')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Q2D_N{N}.png', dpi=300)
        plt.close()


def main():
    data = load_coin_data('data.txt')
    problem_2b(data)
    problem_2c(data)
    problem_2d(data)


if __name__ == '__main__':
    main()
