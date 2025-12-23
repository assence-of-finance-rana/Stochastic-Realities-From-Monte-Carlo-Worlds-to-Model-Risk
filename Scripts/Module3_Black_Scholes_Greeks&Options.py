# %%
import numpy as np
import matplotlib.pyplot as plt

# Stock prices at maturity
S_T = np.linspace(0, 200, 500)
K = 100  # strike price

# Call option payoff
call_payoff = np.maximum(S_T - K, 0)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(S_T, call_payoff, linewidth=2)
plt.axvline(K, linestyle="--", alpha=0.6)
plt.xlabel("Stock Price at Maturity $S_T$")
plt.ylabel("Call Option Payoff")
plt.title("European Call Option Payoff Function")
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100
mu = 0.08
sigma = 0.2
T = 1.0
dt = 1/252
n_steps = int(T/dt)
n_paths = 20

# Time grid
t = np.linspace(0, T, n_steps)

# Simulate paths
np.random.seed(42)
paths = np.zeros((n_steps, n_paths))
paths[0] = S0

for i in range(1, n_steps):
    Z = np.random.randn(n_paths)
    paths[i] = paths[i-1] * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    )

# Plot
plt.figure(figsize=(8, 4))
for i in range(n_paths):
    plt.plot(t, paths[:, i], alpha=0.7)

plt.xlabel("Time (years)")
plt.ylabel("Stock Price")
plt.title("Geometric Brownian Motion Stock Paths")
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0
n_sim = 100_000

np.random.seed(0)

# Risk-neutral simulation
Z = np.random.randn(n_sim)
ST = S0 * np.exp(
    (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z
)

# Option payoff
payoff = np.maximum(ST - K, 0)

# Discounted price
call_price = np.exp(-r * T) * payoff.mean()

call_price


# %%
plt.figure(figsize=(7,4))
plt.hist(ST, bins=100, density=True, alpha=0.7)
plt.axvline(K, color="red", linestyle="--", label="Strike")
plt.xlabel("Stock Price at Maturity")
plt.ylabel("Density")
plt.title("Risk-Neutral Distribution of $S_T$")
plt.legend()
plt.grid(True)
plt.show()


# %%
plt.figure(figsize=(7,4))
plt.hist(payoff, bins=100, density=True, alpha=0.7)
plt.xlabel("Call Payoff")
plt.ylabel("Density")
plt.title("Distribution of Call Option Payoff")
plt.grid(True)
plt.show()


# %%
discounted_payoff = np.exp(-r * T) * payoff

plt.figure(figsize=(7,4))
plt.hist(discounted_payoff, bins=100, density=True, alpha=0.7)
plt.xlabel("Discounted Payoff")
plt.ylabel("Density")
plt.title("Discounted Payoff Distribution (Present Value)")
plt.grid(True)
plt.show()


# %%
import numpy as np
from scipy.stats import norm

def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


# %%
def monte_carlo_call(S0, K, r, sigma, T, n_sim, seed=0):
    np.random.seed(seed)
    Z = np.random.randn(n_sim)
    ST = S0 * np.exp(
        (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z
    )
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * payoff.mean()


# %%
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1

bs_price = black_scholes_call(S0, K, r, sigma, T)
mc_price = monte_carlo_call(S0, K, r, sigma, T, n_sim=200_000)

bs_price, mc_price


# %%
import matplotlib.pyplot as plt

sim_sizes = np.logspace(3, 6, 12, dtype=int)
mc_prices = []

for n in sim_sizes:
    price = monte_carlo_call(S0, K, r, sigma, T, n_sim=n)
    mc_prices.append(price)

plt.figure(figsize=(7,4))
plt.plot(sim_sizes, mc_prices, marker="o", label="Monte Carlo Price")
plt.axhline(bs_price, color="red", linestyle="--", label="Black-Scholes Price")
plt.xscale("log")
plt.xlabel("Number of Simulations (log scale)")
plt.ylabel("Option Price")
plt.title("Monte Carlo Convergence to Black-Scholes")
plt.legend()
plt.grid(True)
plt.show()


# %%
error = np.abs(np.array(mc_prices) - bs_price)

plt.figure(figsize=(7,4))
plt.plot(sim_sizes, error, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Simulations")
plt.ylabel("Absolute Error")
plt.title("Monte Carlo Error Decay")
plt.grid(True)
plt.show()


# %%
vols = np.linspace(0.05, 0.6, 30)
prices = [black_scholes_call(S0, K, r, v, T) for v in vols]

plt.figure(figsize=(7,4))
plt.plot(vols, prices)
plt.xlabel("Volatility")
plt.ylabel("Call Price")
plt.title("Option Price vs Volatility (Convexity)")
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

S_range = np.linspace(50, 150, 40)
sigma_range = np.linspace(0.05, 0.6, 40)

S_grid, sigma_grid = np.meshgrid(S_range, sigma_range)

prices = black_scholes_call(S_grid, K=100, r=0.05, sigma=sigma_grid, T=1)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, sigma_grid, prices, cmap='viridis')

ax.set_xlabel("Stock Price $S_0$")
ax.set_ylabel("Volatility $\\sigma$")
ax.set_zlabel("Call Price")
ax.set_title("Black–Scholes Call Price Surface")

plt.show()


# %%
def monte_carlo_call(S0, K, r, sigma, T, n_sim=50_000):
    Z = np.random.randn(n_sim)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * payoff.mean()

mc_prices = np.zeros_like(S_grid)

for i in range(S_grid.shape[0]):
    for j in range(S_grid.shape[1]):
        mc_prices[i, j] = monte_carlo_call(
            S_grid[i, j], 100, 0.05, sigma_grid[i, j], 1
        )

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, sigma_grid, mc_prices, cmap='plasma')

ax.set_xlabel("Stock Price $S_0$")
ax.set_ylabel("Volatility $\\sigma$")
ax.set_zlabel("Monte Carlo Price")
ax.set_title("Monte Carlo Option Price Surface")

plt.show()


# %%
error_surface = np.abs(mc_prices - prices)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, sigma_grid, error_surface, cmap='inferno')

ax.set_xlabel("Stock Price $S_0$")
ax.set_ylabel("Volatility $\\sigma$")
ax.set_zlabel("Absolute Error")
ax.set_title("Monte Carlo vs Black–Scholes Error Surface")

plt.show()


# %%
from scipy.stats import norm
import numpy as np

def bs_delta_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)


# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

S = np.linspace(50, 150, 40)
sigma = np.linspace(0.05, 0.6, 40)
Sg, sg = np.meshgrid(S, sigma)

delta_surface = bs_delta_call(Sg, 100, 0.05, sg, 1)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Sg, sg, delta_surface, cmap='viridis')

ax.set_xlabel("Stock Price $S_0$")
ax.set_ylabel("Volatility $\\sigma$")
ax.set_zlabel("Delta")
ax.set_title("Delta Surface (Directional Risk)")
plt.show()


# %%
def bs_gamma(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S0 * sigma * np.sqrt(T))


# %%
gamma_surface = bs_gamma(Sg, 100, 0.05, sg, 1)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Sg, sg, gamma_surface, cmap='inferno')

ax.set_xlabel("Stock Price $S_0$")
ax.set_ylabel("Volatility $\\sigma$")
ax.set_zlabel("Gamma")
ax.set_title("Gamma Surface (Convexity Risk)")
plt.show()


# %%
def bs_vega(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S0 * norm.pdf(d1) * np.sqrt(T)


# %%
vega_surface = bs_vega(Sg, 100, 0.05, sg, 1)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Sg, sg, vega_surface, cmap='plasma')

ax.set_xlabel("Stock Price $S_0$")
ax.set_ylabel("Volatility $\\sigma$")
ax.set_zlabel("Vega")
ax.set_title("Vega Surface (Volatility Risk)")
plt.show()


# %%
# ================================
# FINAL MODULE 3 — OPTION RISK MANIFOLD
# ================================

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# ---------- Black-Scholes Functions ----------
def black_scholes_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_gamma(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# ---------- Grid ----------
S = np.linspace(50, 150, 60)
sigma = np.linspace(0.08, 0.6, 60)
Sg, sg = np.meshgrid(S, sigma)

K, r, T = 100, 0.05, 1.0

price = black_scholes_call(Sg, K, r, sg, T)
delta = bs_delta(Sg, K, r, sg, T)
gamma = bs_gamma(Sg, K, r, sg, T)

# ---------- Base Surface (Price + Delta Color) ----------
fig = go.Figure()

fig.add_surface(
    x=Sg,
    y=sg,
    z=price,
    surfacecolor=delta,
    colorscale="Viridis",
    colorbar=dict(title="Delta"),
    opacity=0.95,
    name="Option Price Surface"
)

# ---------- Gamma Singularity Ridge ----------
gamma_cut = np.percentile(gamma, 95)
mask = gamma > gamma_cut

fig.add_scatter3d(
    x=Sg[mask],
    y=sg[mask],
    z=price[mask],
    mode="markers",
    marker=dict(size=3, color="red"),
    name="Gamma Singularity"
)

# ---------- Vega Curtains ----------
for vol_slice in [0.2, 0.35, 0.5]:
    idx = np.abs(sigma - vol_slice).argmin()
    fig.add_surface(
        x=Sg[idx:idx+1],
        y=sg[idx:idx+1],
        z=price[idx:idx+1],
        colorscale="Plasma",
        opacity=0.4,
        showscale=False
    )

# ---------- Monte Carlo Ghost Paths ----------
np.random.seed(0)
S0, sig0 = 100, 0.25

for _ in range(20):
    Z = np.random.randn(252)
    ST = S0 * np.exp(np.cumsum((r - 0.5 * sig0**2)/252 + sig0*np.sqrt(1/252)*Z))
    sig_path = np.linspace(0.15, 0.45, len(ST))
    price_path = black_scholes_call(ST, K, r, sig_path, T)

    fig.add_scatter3d(
        x=ST,
        y=sig_path,
        z=price_path,
        mode="lines",
        line=dict(color="white", width=1),
        opacity=0.25,
        showlegend=False
    )

# ---------- Stress Shock Surface ----------
stress_price = black_scholes_call(Sg, K, r, sg * 1.4, T)

fig.add_surface(
    x=Sg,
    y=sg,
    z=stress_price,
    colorscale="Reds",
    opacity=0.25,
    showscale=False,
    name="Volatility Shock"
)

# ---------- Layout ----------
fig.update_layout(
    title="Option Risk Manifold under Volatility Stress<br>"
          "<sub>Height: Price | Color: Delta | Red Ridge: Gamma | Curtains: Vega | Paths: Monte Carlo</sub>",
    scene=dict(
        xaxis_title="Stock Price",
        yaxis_title="Volatility",
        zaxis_title="Option Price"
    ),
    width=1200,
    height=850
)

fig.show()


# %%



