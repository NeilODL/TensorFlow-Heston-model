import numpy as np
import matplotlib.pyplot as plt
from math import sinh, asinh

def make_grid(ns, s_min, s_max, c, nv, v_min, v_max, d):
    delta_zeta_i = (1.0 / ns) * (asinh((s_max - s_min) / c))
    zeta_s = [asinh((s_min - s_min) / c) + i * delta_zeta_i for i in range(ns + 1)]
    vec_s = [g(zeta_s[i], s_min, c) for i in range(ns + 1)]

    delta_eta = (1.0 / nv) * (asinh((v_max - v_min) / d))
    eta_v = [asinh((v_min - v_min) / d) + i * delta_eta for i in range(nv + 1)]
    vec_v = [h(eta_v[i], v_min, d) for i in range(nv + 1)]

    x, y = np.meshgrid(vec_s, vec_v)
    
    return x, y

def g(xi, s_min, c):
    return s_min + c * sinh(xi)

def h(xi, v_min, d):
    return v_min + d * sinh(xi)

def plot_non_uniform_grid():
    # Parameters for the grid
    ns = 50  # Number of price steps
    nv = 25  # Number of variance steps
    S_max = 8 * 1.0 * (1 + 0.5 * 0.25)  # Example S_max
    S_min = 1.0 * (1 - 0.5 * 0.25)      # Example S_min
    V_max = 5.0 * (1 + 0.5 * 0.25)      # Example V_max
    V_min = max(0.0, 0.5 - 0.5 * 0.25)  # Example V_min
    c = (S_max - S_min) / 10
    d = (V_max - V_min) / 10

    # Generate the non-uniform grid
    x, y = make_grid(ns, S_min, S_max, c, nv, V_min, V_max, d)

    # Plot the non-uniform grid
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='green', s=10, alpha=0.7, label="Grid Points")
    plt.title("Non-Uniform Grid Used in the FDM Model")
    plt.xlabel("Underlying Asset Price (S)")
    plt.ylabel("Variance (V)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_non_uniform_grid()
