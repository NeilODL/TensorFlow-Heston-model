import numpy as np
from scipy.integrate import quad
from scipy.stats import qmc
import csv
import time  # Importing the time module

def heston_char_function(u, t, S0, r, kappa, theta, sigma, rho, v0):
    """
    Heston characteristic function.
    """
    i = complex(0, 1)
    a = kappa * theta

    d = np.sqrt((rho * sigma * i * u - kappa)**2 + (sigma**2) * (i * u + u**2))
    g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d)

    C = (r * i * u * t) + (a / sigma**2) * ((kappa - rho * sigma * i * u - d) * t -
        2 * np.log((1 - g * np.exp(-d * t)) / (1 - g)))

    D = ((kappa - rho * sigma * i * u - d) / sigma**2) * (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))

    phi = np.exp(C + D * v0 + i * u * np.log(S0))
    return phi

def heston_p1_integrand(u, S0, K, t, r, kappa, theta, sigma, rho, v0):
    """
    Integrand for computing P1 in the Heston model.
    """
    i = complex(0, 1)
    phi = heston_char_function(u - i, t, S0, r, kappa, theta, sigma, rho, v0)
    phi_minus_i = heston_char_function(-i, t, S0, r, kappa, theta, sigma, rho, v0)
    numerator = np.exp(-i * u * np.log(K)) * phi
    denominator = i * u * phi_minus_i
    integrand = (numerator / denominator).real
    return integrand

def heston_p2_integrand(u, S0, K, t, r, kappa, theta, sigma, rho, v0):
    """
    Integrand for computing P2 in the Heston model.
    """
    i = complex(0, 1)
    phi = heston_char_function(u, t, S0, r, kappa, theta, sigma, rho, v0)
    numerator = np.exp(-i * u * np.log(K)) * phi
    denominator = i * u
    integrand = (numerator / denominator).real
    return integrand

def heston_call_price(S0, K, t, r, kappa, theta, sigma, rho, v0):
    """
    Computes the European call option price using the Heston model via Fourier inversion.
    """
    # Compute P1
    integrand_p1 = lambda u: heston_p1_integrand(u, S0, K, t, r, kappa, theta, sigma, rho, v0)
    integral_p1, _ = quad(integrand_p1, 0, 100, limit=100, epsabs=1e-8, epsrel=1e-8)
    P1 = 0.5 + (1 / np.pi) * integral_p1

    # Compute P2
    integrand_p2 = lambda u: heston_p2_integrand(u, S0, K, t, r, kappa, theta, sigma, rho, v0)
    integral_p2, _ = quad(integrand_p2, 0, 100, limit=100, epsabs=1e-8, epsrel=1e-8)
    P2 = 0.5 + (1 / np.pi) * integral_p2

    # Compute the call price
    call_price = S0 * P1 - K * np.exp(-r * t) * P2
    return call_price

def generate_lhs_samples(samples=100):
    """
    Generates Latin Hypercube Sampling for the eight Heston model parameters using scipy's LHS.
    """
    lhs_sampler = qmc.LatinHypercube(d=8)
    lhs_samples = lhs_sampler.random(n=samples)

    # Define parameter ranges
    m_min, m_max = 0.6, 1.4
    tau_min, tau_max = 0.1, 1.4  # Time to maturity in years
    r_min, r_max = 0.0, 0.10  # Risk-free rate
    rho_min, rho_max = -0.95, 0.0  # Correlation
    kappa_min, kappa_max = 0.0, 2.0  # Reversion speed
    theta_min, theta_max = 0.0, 0.5  # Long average variance
    sigma_min, sigma_max = 0.0, 0.5  # Volatility of volatility
    v0_min, v0_max = 0.05, 0.5  # Initial variance

    # Scale LHS samples to the parameter ranges
    m = m_min + lhs_samples[:, 0] * (m_max - m_min)
    tau = tau_min + lhs_samples[:, 1] * (tau_max - tau_min)
    r = r_min + lhs_samples[:, 2] * (r_max - r_min)
    rho = rho_min + lhs_samples[:, 3] * (rho_max - rho_min)
    kappa = kappa_min + lhs_samples[:, 4] * (kappa_max - kappa_min)
    theta = theta_min + lhs_samples[:, 5] * (theta_max - theta_min)
    sigma = sigma_min + lhs_samples[:, 6] * (sigma_max - sigma_min)
    v0 = v0_min + lhs_samples[:, 7] * (v0_max - v0_min)

    return np.column_stack((m, tau, r, rho, kappa, theta, sigma, v0))

def save_to_csv(data, filename="Fourier_Prices.csv"):
    """
    Saves the list of call prices and parameter sets to a CSV file.
    """
    header = ['moneyness', 'tau', 'r', 'rho', 'kappa', 'theta', 'sigma', 'initial_variance', 'option_price']

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write header
        for row in data:
            writer.writerow(row)

def main():
    N = int(input("Enter the number of prices to generate (N): "))  # User-defined N

    start_time = time.time()  # Start time for timing the function

    # Generate N parameter sets using Latin Hypercube Sampling
    param_sets = generate_lhs_samples(N)

    S0 = 1.0  # Fixed initial stock price
    result_data = []

    for idx, params in enumerate(param_sets):
        m, tau, r, rho, kappa, theta, sigma, v0 = params
        K = S0 / m  # Compute strike price based on moneyness

        # Compute the European call option price using the Heston model
        price = heston_call_price(S0, K, tau, r, kappa, theta, sigma, rho, v0)

        # Apply the same filtering as in the FDM script
        if (0.6 <= m <= 1.4) and (0.05 <= v0 <= 0.5):
            result_data.append([m, tau, r, rho, kappa, theta, sigma, v0, price])

        # Optional: Print progress every 1000 iterations
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{N} parameter sets.")

    # Save the simulated data to a CSV file
    save_to_csv(result_data, "Fourier_Prices.csv")

    # Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution Time: {execution_time:.4f} seconds")
    print(f"\nSimulated data for {len(result_data)} valid prices have been saved to Fourier_Prices.csv.")

if __name__ == "__main__":
    main()
