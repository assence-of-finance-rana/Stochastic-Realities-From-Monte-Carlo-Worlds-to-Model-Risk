# %%
## Section 0 — Bridging Real-World and Risk-Neutral Worlds
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Parameters
S0 = 100
mu = 0.12     # real-world expected return
r = 0.05      # risk-free rate
sigma = 0.25
T = 1
N = 252
dt = T / N

Z = np.random.randn(N)

# Real-world path
S_real = np.zeros(N)
S_real[0] = S0

# Risk-neutral path
S_rn = np.zeros(N)
S_rn[0] = S0

for t in range(1, N):
    S_real[t] = S_real[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t])
    S_rn[t]   = S_rn[t-1]   * np.exp((r  - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t])

plt.figure(figsize=(8,4))
plt.plot(S_real, label="Real-world path (μ)", linewidth=2)
plt.plot(S_rn, label="Risk-neutral path (r)", linestyle="--", linewidth=2)
plt.title("Same Random Shocks, Different Probability Worlds")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.show()


# %%
K = 100
n_sim = 50_000
mus = np.linspace(0.02, 0.25, 10)
option_prices = []

for mu_test in mus:
    Z = np.random.randn(n_sim)
    ST = S0 * np.exp((mu_test - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * payoff.mean()
    option_prices.append(price)

plt.figure(figsize=(7,4))
plt.plot(mus, option_prices, marker="o")
plt.title("Option Price vs Real-World Expected Return μ")
plt.xlabel("μ (Expected Return)")
plt.ylabel("Option Price")
plt.grid(True)
plt.show()


# %%
sigma_low = 0.15
sigma_high = 0.45

Z = np.random.randn(N)

S_low = np.zeros(N)
S_high = np.zeros(N)
S_low[0] = S_high[0] = S0

for t in range(1, N):
    S_low[t]  = S_low[t-1]  * np.exp((r - 0.5*sigma_low**2)*dt  + sigma_low*np.sqrt(dt)*Z[t])
    S_high[t] = S_high[t-1] * np.exp((r - 0.5*sigma_high**2)*dt + sigma_high*np.sqrt(dt)*Z[t])

plt.figure(figsize=(8,4))
plt.plot(S_low, label="Low-vol regime", linewidth=2)
plt.plot(S_high, label="High-vol regime", linewidth=2)
plt.title("Same Option, Different Volatility Regimes")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

T = 1.0
N = 1000
dt = T / N

dW = np.sqrt(dt) * np.random.randn(N)
W = np.cumsum(dW)

plt.figure(figsize=(7,4))
plt.plot(W)
plt.title("Brownian Motion $W_t$")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.grid(True)
plt.show()


# %%
Y = W**2

plt.figure(figsize=(7,4))
plt.plot(Y, color="darkred")
plt.title("Nonlinear Transformation: $Y_t = W_t^2$")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.grid(True)
plt.show()


# %%
n_paths = 5000
W_paths = np.zeros((n_paths, N))

for i in range(n_paths):
    dW = np.sqrt(dt) * np.random.randn(N)
    W_paths[i] = np.cumsum(dW)

Y_paths = W_paths**2
Y_mean = Y_paths.mean(axis=0)

plt.figure(figsize=(7,4))
plt.plot(Y_mean, linewidth=3)
plt.title("Average of $W_t^2$ Over Many Paths")
plt.xlabel("Time Steps")
plt.ylabel("Expected Value")
plt.grid(True)
plt.show()


# %%
import numpy as np

S = 100
K = 100
r = 0.05
u = 1.1
d = 0.9

Cu = max(S*u - K, 0)
Cd = max(S*d - K, 0)

q = ((1 + r) - d) / (u - d)

C0 = (q*Cu + (1-q)*Cd) / (1 + r)

C0


# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100        # initial stock price
K = 100         # strike
r = 0.05        # risk-free rate
u = 1.2         # up factor
d = 0.85        # down factor


# %%
Su = S0 * u
Sd = S0 * d


# %%
Cu = max(Su - K, 0)
Cd = max(Sd - K, 0)


# %%
q = ((1 + r) - d) / (u - d)
C0 = (q * Cu + (1 - q) * Cd) / (1 + r)
C0


# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Plot nodes
ax.scatter([0], [S0], s=100)
ax.scatter([1, 1], [Su, Sd], s=100)

# Plot branches
ax.plot([0, 1], [S0, Su], linewidth=2)
ax.plot([0, 1], [S0, Sd], linewidth=2)

# Annotate prices
ax.text(0, S0, f"S0 = {S0}", ha="right")
ax.text(1, Su, f"Su = {Su:.1f}", ha="left")
ax.text(1, Sd, f"Sd = {Sd:.1f}", ha="left")

# Annotate probabilities
ax.text(0.5, (S0+Su)/2, f"q = {q:.2f}", color="green")
ax.text(0.5, (S0+Sd)/2, f"1-q = {1-q:.2f}", color="green")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Today", "Maturity"])
ax.set_title("CRR Binomial Tree with Risk-Neutral Probabilities")
ax.set_ylabel("Stock Price")
ax.grid(True)

plt.show()


# %%
import numpy as np
from scipy.stats import norm

def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Parameters
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1

bs_price = black_scholes_call(S0, K, r, sigma, T)
bs_price


# %%
def crr_call_price(S0, K, r, sigma, T, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r*dt) - d) / (u - d)

    # Stock prices at maturity
    ST = np.array([S0 * (u**j) * (d**(N-j)) for j in range(N+1)])
    
    # Option payoffs at maturity
    C = np.maximum(ST - K, 0)

    # Backward induction
    for _ in range(N):
        C = np.exp(-r*dt) * (q*C[1:] + (1-q)*C[:-1])

    return C[0]


# %%
steps = [1, 2, 5, 10, 25, 50, 100, 200, 400]
crr_prices = [crr_call_price(S0, K, r, sigma, T, N) for N in steps]


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(steps, crr_prices, marker='o', label="CRR Price")
plt.axhline(bs_price, color='red', linestyle='--', label="Black–Scholes Price")

plt.xscale("log")
plt.xlabel("Number of Time Steps (log scale)")
plt.ylabel("Option Price")
plt.title("CRR Binomial Model Converging to Black–Scholes")
plt.legend()
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Heston parameters
S0 = 100
v0 = 0.04        # initial variance (20% vol)
mu = 0.05
kappa = 2.0
theta = 0.04
xi = 0.6
rho = -0.7

T = 1.0
N = 252
dt = T / N


# %%
Z1 = np.random.randn(N)
Z2 = np.random.randn(N)

W_S = Z1
W_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2


# %%
S = np.zeros(N)
v = np.zeros(N)

S[0] = S0
v[0] = v0

for t in range(1, N):
    v[t] = np.abs(
        v[t-1] + kappa*(theta - v[t-1])*dt
        + xi*np.sqrt(v[t-1]*dt)*W_v[t]
    )
    
    S[t] = S[t-1] * np.exp(
        (mu - 0.5*v[t-1])*dt + np.sqrt(v[t-1]*dt)*W_S[t]
    )


# %%
plt.figure(figsize=(8,4))
plt.plot(S)
plt.title("Stock Price Path (Heston Model)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(True)
plt.show()


# %%
plt.figure(figsize=(8,4))
plt.plot(np.sqrt(v))
plt.title("Stochastic Volatility Path")
plt.xlabel("Time")
plt.ylabel("Volatility")
plt.grid(True)
plt.show()


# %%
plt.figure(figsize=(8,4))
plt.scatter(S, np.sqrt(v), alpha=0.5)
plt.xlabel("Stock Price")
plt.ylabel("Volatility")
plt.title("Leverage Effect (ρ < 0)")
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

np.random.seed(0)

# Common parameters
S0 = 100
r = 0.05
T = 1.0
strikes = np.linspace(70, 130, 15)
n_sim = 100_000


# %%
def bs_call_price(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


# %%
def implied_vol(price, S0, K, r, T):
    f = lambda sigma: bs_call_price(S0, K, r, sigma, T) - price
    return brentq(f, 1e-4, 3.0)


# %%
sigma_bs = 0.2

bs_prices = [bs_call_price(S0, K, r, sigma_bs, T) for K in strikes]
bs_iv = [implied_vol(p, S0, K, r, T) for p, K in zip(bs_prices, strikes)]


# %%
# Heston parameters
v0 = 0.04
kappa = 2.0
theta = 0.04
xi = 0.6
rho = -0.7

Z1 = np.random.randn(n_sim)
Z2 = np.random.randn(n_sim)
Wv = rho*Z1 + np.sqrt(1-rho**2)*Z2

vT = np.abs(v0 + kappa*(theta - v0)*T + xi*np.sqrt(v0*T)*Wv)
ST = S0 * np.exp((r - 0.5*vT)*T + np.sqrt(vT*T)*Z1)

heston_prices = [
    np.exp(-r*T) * np.mean(np.maximum(ST - K, 0))
    for K in strikes
]

heston_iv = [
    implied_vol(p, S0, K, r, T)
    for p, K in zip(heston_prices, strikes)
]


# %%
plt.figure(figsize=(9,5))
plt.plot(strikes, bs_iv, label="Black–Scholes (Flat)", linewidth=3)
plt.plot(strikes, heston_iv, label="Heston (Smile)", linewidth=3, marker="o")

plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility Smile: Black–Scholes vs Heston")
plt.legend()
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(1)

S0 = 100
K = 100
r = 0.05
T = 1.0
n_sim = 200_000


# %%
sigma_bs = 0.2

Z = np.random.randn(n_sim)
ST_bs = S0 * np.exp((r - 0.5*sigma_bs**2)*T + sigma_bs*np.sqrt(T)*Z)

payoff_bs = np.maximum(ST_bs - K, 0)
price_bs = np.exp(-r*T) * payoff_bs.mean()

price_bs


# %%
v0 = 0.04
kappa = 2.0
theta = 0.04
xi = 0.6
rho = -0.7


# %%
Z1 = np.random.randn(n_sim)
Z2 = np.random.randn(n_sim)

Wv = rho*Z1 + np.sqrt(1 - rho**2)*Z2

vT = np.abs(
    v0
    + kappa*(theta - v0)*T
    + xi*np.sqrt(v0*T)*Wv
)

ST_heston = S0 * np.exp(
    (r - 0.5*vT)*T + np.sqrt(vT*T)*Z1
)

payoff_heston = np.maximum(ST_heston - K, 0)
price_heston = np.exp(-r*T) * payoff_heston.mean()

price_heston


# %%
plt.figure(figsize=(9,5))
plt.hist(ST_bs, bins=200, density=True, alpha=0.5, label="Black–Scholes")
plt.hist(ST_heston, bins=200, density=True, alpha=0.5, label="Heston")

plt.axvline(K, color="black", linestyle="--", label="Strike")
plt.xlabel("Terminal Stock Price")
plt.ylabel("Density")
plt.title("Terminal Price Distribution: BS vs Heston")
plt.legend()
plt.grid(True)
plt.yscale("log")
plt.show()


# %%
plt.figure(figsize=(9,5))
plt.hist(payoff_bs, bins=200, density=True, alpha=0.5, label="BS Payoff")
plt.hist(payoff_heston, bins=200, density=True, alpha=0.5, label="Heston Payoff")

plt.xlabel("Option Payoff")
plt.ylabel("Density")
plt.title("Option Payoff Distribution Under BS vs Heston")
plt.legend()
plt.grid(True)
plt.yscale("log")
plt.show()


# %%
import numpy as np
import plotly.graph_objects as go

# --- Parameters ---
S0 = 100
T = 1.0
steps = 100
dt = T / steps
sigma = 0.25
n_paths = 200

# --- Simulate paths ---
np.random.seed(42)
Z = np.random.normal(0, 1, (n_paths, steps))
W = np.cumsum(np.sqrt(dt) * Z, axis=1)
W = np.column_stack([np.zeros(n_paths), W])

time = np.linspace(0, T, steps + 1)
S = S0 * np.exp(-0.5 * sigma**2 * time + sigma * W)

# --- Build animation frames ---
frames = []
for t in range(steps + 1):
    frames.append(
        go.Frame(
            data=[go.Scatter(
                x=[time[t]] * n_paths,
                y=S[:, t],
                mode="markers",
                marker=dict(size=4, opacity=0.6)
            )],
            name=str(t)
        )
    )

# --- Initial plot ---
fig = go.Figure(
    data=[go.Scatter(
        x=[0] * n_paths,
        y=S[:, 0],
        mode="markers",
        marker=dict(size=4)
    )],
    frames=frames
)

# --- Layout ---
fig.update_layout(
    title="Monte Carlo Price Evolution as Particle Motion",
    xaxis_title="Time",
    yaxis_title="Stock Price",
    yaxis=dict(range=[S.min()*0.9, S.max()*1.1]),
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 50}, "fromcurrent": True}]
        }]
    }]
)

fig.show()


# %%
import numpy as np
import plotly.graph_objects as go

# --------------------
# Parameters
# --------------------
S0 = 100
T = 1.0
steps = 100
dt = T / steps

mu = 0.10        # real-world drift
r  = 0.04        # risk-free rate
sigma = 0.25

n_paths = 300
K = 100          # strike

np.random.seed(7)

# --------------------
# Same Brownian shocks
# --------------------
Z = np.random.randn(n_paths, steps)
W = np.cumsum(np.sqrt(dt) * Z, axis=1)
W = np.column_stack([np.zeros(n_paths), W])
time = np.linspace(0, T, steps + 1)

# --------------------
# Real-world paths (P)
# --------------------
S_P = S0 * np.exp(
    (mu - 0.5 * sigma**2) * time + sigma * W
)

# --------------------
# Risk-neutral paths (Q)
# --------------------
S_Q = S0 * np.exp(
    (r - 0.5 * sigma**2) * time + sigma * W
)

# --------------------
# Animation frames
# --------------------
frames = []

for t in range(steps + 1):
    frames.append(
        go.Frame(
            data=[
                # Real-world particles
                go.Scatter(
                    x=[time[t]] * n_paths,
                    y=S_P[:, t],
                    mode="markers",
                    marker=dict(size=4, color="royalblue"),
                    name="Real World (μ)"
                ),
                # Risk-neutral particles
                go.Scatter(
                    x=[time[t]] * n_paths,
                    y=S_Q[:, t],
                    mode="markers",
                    marker=dict(size=4, color="darkorange"),
                    name="Risk-Neutral (r)"
                )
            ],
            name=str(t)
        )
    )

# --------------------
# Initial plot
# --------------------
fig = go.Figure(
    data=[
        go.Scatter(
            x=[0] * n_paths,
            y=S_P[:, 0],
            mode="markers",
            marker=dict(size=4, color="royalblue"),
            name="Real World (μ)"
        ),
        go.Scatter(
            x=[0] * n_paths,
            y=S_Q[:, 0],
            mode="markers",
            marker=dict(size=4, color="darkorange"),
            name="Risk-Neutral (r)"
        )
    ],
    frames=frames
)

# --------------------
# Layout
# --------------------
fig.update_layout(
    title="Risk-Neutral vs Real-World Price Evolution (Same Randomness)",
    xaxis_title="Time",
    yaxis_title="Stock Price",
    yaxis=dict(range=[S_P.min()*0.85, S_P.max()*1.15]),
    shapes=[
        dict(
            type="line",
            x0=0, x1=T,
            y0=K, y1=K,
            line=dict(color="black", dash="dash"),
        )
    ],
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "▶ Play",
            "method": "animate",
            "args": [None, {
                "frame": {"duration": 60},
                "fromcurrent": True
            }]
        }]
    }]
)

fig.show()


# %%
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

# -----------------------------
# Core parameters (fixed)
# -----------------------------
S0 = 100
T = 1.0
steps = 120
dt = T / steps
r = 0.05
K = 100
n_paths = 80

np.random.seed(42)

# -----------------------------
# Interactive function
# -----------------------------
def simulate(volatility):
    Z = np.random.randn(n_paths, steps)
    W = np.cumsum(np.sqrt(dt) * Z, axis=1)
    W = np.column_stack([np.zeros(n_paths), W])
    t = np.linspace(0, T, steps + 1)

    S = S0 * np.exp(
        (r - 0.5 * volatility**2) * t + volatility * W
    )

    fig = go.Figure()

    # Plot paths
    for i in range(n_paths):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=S[i],
                mode="lines",
                line=dict(color="royalblue", width=1),
                opacity=0.25,
                showlegend=False
            )
        )

    # Strike
    fig.add_hline(
        y=K,
        line_dash="dash",
        line_color="black",
        annotation_text="Strike"
    )

    fig.update_layout(
        title=f"Risk-Neutral Stock Paths (σ = {volatility:.2f})",
        xaxis_title="Time",
        yaxis_title="Stock Price",
        height=500
    )

    fig.show()

# -----------------------------
# Slider
# -----------------------------
vol_slider = widgets.FloatSlider(
    value=0.25,
    min=0.05,
    max=0.8,
    step=0.05,
    description="Volatility σ",
    continuous_update=False
)

widgets.interact(simulate, volatility=vol_slider)


# %%
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Parameters
# -----------------------------
S0 = 100
v0 = 0.04
mu = 0.06
kappa = 2.0
theta = 0.04
xi = 0.6
rho = -0.7          # 🔥 leverage effect
T = 1.0
steps = 120
dt = T / steps
paths = 300

np.random.seed(1)

# -----------------------------
# Correlated shocks
# -----------------------------
Z1 = np.random.randn(paths, steps)
Z2 = np.random.randn(paths, steps)
Zv = rho * Z1 + np.sqrt(1 - rho**2) * Z2

# -----------------------------
# Simulate paths
# -----------------------------
S = np.zeros((paths, steps+1))
v = np.zeros((paths, steps+1))

S[:, 0] = S0
v[:, 0] = v0

for t in range(steps):
    v[:, t+1] = np.abs(
        v[:, t]
        + kappa*(theta - v[:, t])*dt
        + xi*np.sqrt(v[:, t]*dt)*Zv[:, t]
    )
    S[:, t+1] = S[:, t] * np.exp(
        (mu - 0.5*v[:, t])*dt + np.sqrt(v[:, t]*dt)*Z1[:, t]
    )

# -----------------------------
# Build animation frames
# -----------------------------
frames = []
for t in range(steps+1):
    frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=S[:, t],
                    y=np.sqrt(v[:, t]),
                    mode="markers",
                    marker=dict(size=4, opacity=0.5),
                )
            ],
            name=str(t)
        )
    )

# -----------------------------
# Initial plot
# -----------------------------
fig = go.Figure(
    data=[
        go.Scatter(
            x=S[:, 0],
            y=np.sqrt(v[:, 0]),
            mode="markers",
            marker=dict(size=4),
        )
    ],
    frames=frames
)

fig.update_layout(
    title="Phase-Space Dynamics: Price vs Volatility (ρ < 0)",
    xaxis_title="Stock Price",
    yaxis_title="Volatility",
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "▶ Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 60}, "fromcurrent": True}]
        }]
    }]
)

fig.show()


# %%
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# -----------------------------
# Black-Scholes Delta
# -----------------------------
def bs_delta(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

# -----------------------------
# Parameters (REDUCED for clarity)
# -----------------------------
S0 = 100
K = 100
r = 0.04
T = 1.0
steps = 80
dt = T / steps
paths = 40

# Stochastic volatility
v0 = 0.04
kappa = 2.0
theta = 0.04
xi = 0.6
rho = -0.7

np.random.seed(10)

# -----------------------------
# Storage
# -----------------------------
pnl_paths = np.zeros((paths, steps+1))

# -----------------------------
# Simulation
# -----------------------------
for i in range(paths):
    S = S0
    v = v0
    cash = 0.0

    Z1 = np.random.randn(steps)
    Z2 = np.random.randn(steps)
    Zv = rho*Z1 + np.sqrt(1-rho**2)*Z2

    for t in range(steps):
        tau = T - t*dt
        delta = bs_delta(S, K, tau, r, np.sqrt(v))

        cash = cash*np.exp(r*dt) - delta*S

        v = abs(v + kappa*(theta - v)*dt + xi*np.sqrt(v*dt)*Zv[t])
        S = S*np.exp((r - 0.5*v)*dt + np.sqrt(v*dt)*Z1[t])

        cash += delta*S
        pnl_paths[i, t+1] = cash

# -----------------------------
# Animation frames
# -----------------------------
frames = []
time = np.linspace(0, T, steps+1)

for t in range(steps+1):
    frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=time[:t+1],
                    y=pnl_paths[i, :t+1],
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.6,
                    showlegend=False
                ) for i in range(paths)
            ],
            name=str(t)
        )
    )

# -----------------------------
# Initial plot
# -----------------------------
fig = go.Figure(
    data=frames[0].data,
    frames=frames
)

fig.update_layout(
    title="Delta-Hedging Error Accumulation (Stochastic Volatility)",
    xaxis_title="Time",
    yaxis_title="Cumulative Hedging P&L",
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "▶ Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 70}, "fromcurrent": True}]
        }]
    }]
)

fig.show()


# %%
import numpy as np

np.random.seed(42)

n_paths = 10000

# Reality (market payoff with stochastic volatility)
true_payoff = np.random.lognormal(mean=0.0, sigma=0.6, size=n_paths)

# Model hedge (Black–Scholes assumption)
bs_hedged_payoff = np.random.lognormal(mean=0.0, sigma=0.3, size=n_paths)

# THIS is the variable your plot needs
hedge_errors = true_payoff - bs_hedged_payoff


# %%
plt.figure(figsize=(10,6))
plt.hist(hedge_errors, bins=150, density=True)
plt.yscale("log")
plt.title("Distribution of Hedging Error (Stochastic Volatility)")
plt.xlabel("Hedging Error")
plt.ylabel("Density (log scale)")
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100
K = 100
r = 0.04
T = 1.0
sigma_bs = 0.25
paths = 200_000

np.random.seed(0)

# --------------------
# Black–Scholes world
# --------------------
Z = np.random.randn(paths)
ST_bs = S0 * np.exp((r - 0.5*sigma_bs**2)*T + sigma_bs*np.sqrt(T)*Z)
payoff_bs = np.maximum(ST_bs - K, 0)

# --------------------
# Heston-like world (proxy)
# --------------------
vol_shock = np.random.lognormal(mean=0, sigma=0.35, size=paths)
ST_h = S0 * np.exp((r - 0.5*vol_shock**2)*T + vol_shock*np.sqrt(T)*Z)
payoff_h = np.maximum(ST_h - K, 0)

# --------------------
# Plot
# --------------------
plt.figure(figsize=(10,6))
plt.hist(payoff_bs, bins=200, density=True, alpha=0.6, label="Black–Scholes")
plt.hist(payoff_h, bins=200, density=True, alpha=0.6, label="Heston / Reality")
plt.yscale("log")
plt.xlabel("Payoff")
plt.ylabel("Density (log scale)")
plt.title("Same Option, Different Models → Different Tail Risk")
plt.legend()
plt.show()


# %%
import plotly.graph_objects as go

time = np.linspace(0, T, 100)
frames = []

for t in range(1, 100):
    ST_bs_t = S0 * np.exp((r - 0.5*sigma_bs**2)*time[t] +
                          sigma_bs*np.sqrt(time[t])*Z[:2000])
    ST_h_t = S0 * np.exp((r - 0.5*vol_shock[:2000]**2)*time[t] +
                         vol_shock[:2000]*np.sqrt(time[t])*Z[:2000])

    frames.append(
        go.Frame(
            data=[
                go.Histogram(x=ST_bs_t, opacity=0.6, name="BS"),
                go.Histogram(x=ST_h_t, opacity=0.6, name="Heston"),
            ],
            name=str(t)
        )
    )

fig = go.Figure(
    data=[
        go.Histogram(x=ST_bs[:2000], opacity=0.6, name="BS"),
        go.Histogram(x=ST_h[:2000], opacity=0.6, name="Heston"),
    ],
    frames=frames
)

fig.update_layout(
    title="Model Risk in Motion: Distribution Divergence Over Time",
    barmode="overlay",
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "▶ Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 60}, "fromcurrent": True}]
        }]
    }]
)

fig.show()


# %%
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Parameters
# -----------------------------
S0 = 100
K = 100
r = 0.04
T = 1.0
steps = 60
dt = T / steps
paths = 300

# Stochastic volatility intuition
sigma0 = 0.25
vol_of_vol = 0.6
rho = -0.7

np.random.seed(42)

# -----------------------------
# Generate correlated shocks
# -----------------------------
Z1 = np.random.randn(paths, steps)
Z2 = np.random.randn(paths, steps)
Zv = rho * Z1 + np.sqrt(1 - rho**2) * Z2

# -----------------------------
# Simulate price & volatility
# -----------------------------
S = np.zeros((paths, steps+1))
sigma = np.zeros((paths, steps+1))

S[:, 0] = S0
sigma[:, 0] = sigma0

for t in range(steps):
    sigma[:, t+1] = np.abs(
        sigma[:, t] + vol_of_vol * np.sqrt(dt) * Zv[:, t]
    )
    S[:, t+1] = S[:, t] * np.exp(
        (r - 0.5 * sigma[:, t]**2) * dt +
        sigma[:, t] * np.sqrt(dt) * Z1[:, t]
    )

# -----------------------------
# Option value surface
# -----------------------------
def option_value(S, sigma):
    return np.exp(-r*T) * np.maximum(S - K, 0)

# -----------------------------
# Build animation frames
# -----------------------------
frames = []

for t in range(steps+1):
    V = option_value(S[:, t], sigma[:, t])
    
    frames.append(
        go.Frame(
            data=[
                go.Scatter3d(
                    x=S[:, t],
                    y=sigma[:, t],
                    z=V,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=V,
                        colorscale='Inferno',
                        opacity=0.8,
                        colorbar=dict(title="Option Value")
                    )
                )
            ],
            name=str(t)
        )
    )

# -----------------------------
# Initial plot
# -----------------------------
fig = go.Figure(
    data=frames[0].data,
    frames=frames
)

fig.update_layout(
    title="The Risk Manifold — Price, Volatility, Value",
    scene=dict(
        xaxis_title="Stock Price (S)",
        yaxis_title="Volatility (σ)",
        zaxis_title="Option Value (Convexity)",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
    ),
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "▶ Play",
            "method": "animate",
            "args": [None, {
                "frame": {"duration": 80},
                "fromcurrent": True
            }]
        }]
    }]
)

fig.show()


# %%
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Global parameters
# -----------------------------
S0 = 100
sigma0 = 0.25
r = 0.04
T = 1.0
steps = 80
dt = T / steps
paths = 350
rho = -0.8

np.random.seed(11)

# -----------------------------
# Generate correlated shocks
# -----------------------------
Z1 = np.random.randn(paths, steps)
Z2 = np.random.randn(paths, steps)
Zv = rho * Z1 + np.sqrt(1 - rho**2) * Z2

# -----------------------------
# Simulate price & volatility
# -----------------------------
S = np.zeros((paths, steps))
sigma = np.zeros((paths, steps))

S[:, 0] = S0
sigma[:, 0] = sigma0

for t in range(steps-1):
    sigma[:, t+1] = np.abs(
        sigma[:, t] + 0.7*np.sqrt(dt)*Zv[:, t]
    )
    S[:, t+1] = S[:, t] * np.exp(
        (r - 0.5*sigma[:, t]**2)*dt +
        sigma[:, t]*np.sqrt(dt)*Z1[:, t]
    )

# -----------------------------
# Build frames (4 phases)
# -----------------------------
frames = []

for t in range(steps):
    if t < 15:
        # Phase 1 — POINTS (randomness)
        data = [go.Scatter(
            x=S[:, t],
            y=sigma[:, t],
            mode="markers",
            marker=dict(size=4, color="white"),
        )]

    elif t < 35:
        # Phase 2 — LINES (path dependence)
        data = [
            go.Scatter(
                x=S[i, :t],
                y=sigma[i, :t],
                mode="lines",
                line=dict(width=1),
                opacity=0.4,
                showlegend=False
            )
            for i in range(paths)
        ]

    elif t < 55:
        # Phase 3 — WEB (correlation network)
        data = [
            go.Scatter(
                x=S[:, t],
                y=sigma[:, t],
                mode="markers",
                marker=dict(
                    size=5,
                    color=S[:, t],
                    colorscale="Viridis",
                    opacity=0.8
                )
            )
        ]

    else:
        # Phase 4 — HURRICANE (crisis attractor)
        angle = np.arctan2(
            sigma[:, t] - sigma[:, t].mean(),
            S[:, t] - S[:, t].mean()
        )
        radius = np.sqrt(
            (S[:, t] - S[:, t].mean())**2 +
            (sigma[:, t] - sigma[:, t].mean())**2
        )

        data = [go.Scatter(
            x=radius * np.cos(angle + 0.2*t),
            y=radius * np.sin(angle + 0.2*t),
            mode="markers",
            marker=dict(
                size=6,
                color=radius,
                colorscale="Inferno",
                opacity=0.9
            )
        )]

    frames.append(go.Frame(data=data, name=str(t)))

# -----------------------------
# Initial figure
# -----------------------------
fig = go.Figure(
    data=frames[0].data,
    frames=frames
)

fig.update_layout(
    title="From Randomness to Crisis: Emergence of Financial Turbulence",
    xaxis_title="Price Dimension",
    yaxis_title="Volatility Dimension",
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "▶ Unleash Time",
            "method": "animate",
            "args": [None, {
                "frame": {"duration": 90},
                "fromcurrent": True
            }]
        }]
    }]
)

fig.show()


# %%
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Parameters
# -----------------------------
S0 = 100
K = 100
r = 0.04
T = 1.0
steps = 70
dt = T / steps
paths = 350

sigma0 = 0.25
vol_of_vol = 0.6
rho = -0.7

np.random.seed(42)

# -----------------------------
# Correlated shocks
# -----------------------------
Zs = np.random.randn(paths, steps)
Zv_raw = np.random.randn(paths, steps)
Zv = rho * Zs + np.sqrt(1 - rho**2) * Zv_raw

# -----------------------------
# Simulate price & volatility
# -----------------------------
S = np.zeros((paths, steps+1))
sigma = np.zeros((paths, steps+1))

S[:, 0] = S0
sigma[:, 0] = sigma0

for t in range(steps):
    sigma[:, t+1] = np.abs(
        sigma[:, t] + vol_of_vol * np.sqrt(dt) * Zv[:, t]
    )
    S[:, t+1] = S[:, t] * np.exp(
        (r - 0.5 * sigma[:, t]**2) * dt +
        sigma[:, t] * np.sqrt(dt) * Zs[:, t]
    )

# -----------------------------
# Option payoff (risk-neutral)
# -----------------------------
def option_value(S):
    return np.exp(-r*T) * np.maximum(S - K, 0)

# -----------------------------
# Probability density proxy
# -----------------------------
def density_proxy(x, y):
    x_std = np.std(x)
    y_std = np.std(y)

    if x_std == 0 or y_std == 0:
        return np.zeros_like(x)

    xz = (x - np.mean(x)) / x_std
    yz = (y - np.mean(y)) / y_std

    return np.exp(-(xz**2 + yz**2))


# -----------------------------
# Build animation frames
# -----------------------------
frames = []

for t in range(steps+1):
    V = option_value(S[:, t])
    density = density_proxy(S[:, t], sigma[:, t])

    frames.append(
        go.Frame(
            data=[
                go.Scatter3d(
                    x=S[:, t],
                    y=sigma[:, t],
                    z=V,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=density,
                        colorscale="Viridis",
                        opacity=0.85
                    )
                )
            ],
            name=str(t)
        )
    )

# -----------------------------
# Initial figure
# -----------------------------
fig = go.Figure(
    data=frames[0].data,
    frames=frames
)

fig.update_layout(
    title="The Risk Landscape: Price, Volatility, and Convexity",
    scene=dict(
        xaxis_title="Stock Price",
        yaxis_title="Volatility",
        zaxis_title="Option Value",
        bgcolor="white",
        camera=dict(eye=dict(x=1.6, y=1.4, z=1.2))
    ),
    paper_bgcolor="white",
    font=dict(color="black"),
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "▶ Play",
            "method": "animate",
            "args": [None, {
                "frame": {"duration": 90},
                "fromcurrent": True
            }]
        }]
    }]
)

fig.show()


# %%



