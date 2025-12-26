# Stochastic Realities: From Monte Carlo Worlds to Model Risk

This project is a deep, end-to-end exploration of **financial randomness, pricing models, and model risk**, built progressively from first principles to advanced stochastic volatility frameworks.

Rather than treating Monte Carlo, Markov chains, or Black–Scholes as isolated tools, this project **connects them into a single evolving narrative**:  
how assumptions shape markets, how models diverge from reality, and why hedging errors are unavoidable when volatility itself is random.

The project is structured into **four conceptual modules**, each with its own mathematical, computational, and financial focus.

---
## Project Structure

```text
MonteCarlo_Portfolio_Project/
├── Scripts/
│   ├── Module1_MonteCarlo_from_scratch.py
│   ├── Module2_Regime_Switching_MonteCarlo.py
│   ├── Module3_Black_Scholes_Greeks&Options.py
│   └── Module4_Model_Risk_Hedging_Errors.py
│
├── Outputs/
│   ├── Figures_Monte_Carlo/
│   ├── Figures_Markov_Chains/
│   ├── Figures_Black_Scholes/
│   └── Figures_Beyond_Black_Scholes/



---

## Module 1 — Monte Carlo From First Principles

**Central idea:**  
Monte Carlo is not a pricing method — it is a **probability generator**.

This module builds Monte Carlo simulations from scratch using Geometric Brownian Motion and shows how thousands of random paths form statistical structure.

Key outcomes:
- Asset prices are treated as stochastic processes, not forecasts
- Individual paths are meaningless; distributions are everything
- Randomness aggregates into shape, skew, and tail risk

**Insight:**  
Before finance, before models, before prices — there is uncertainty.  
Monte Carlo is the language of that uncertainty.

---

## Module 2 — Regime Switching & Markov Dynamics

**Central idea:**  
Markets are not governed by one volatility — they move through regimes.

This module introduces discrete market states (e.g., calm, volatile, crisis) and uses Markov chains to model transitions between them.

Key outcomes:
- Volatility clustering emerges naturally
- Risk becomes path-dependent through regime memory
- Identical shocks have different effects depending on state

**Insight:**  
Markets remember their past — not through prices, but through regimes.

---

## Module 3 — Black-Scholes, Greeks & the Illusion of Control

**Central idea:**  
Black–Scholes is mathematically elegant and empirically fragile.

This module implements Black–Scholes pricing and Greeks, then studies their implications under Monte Carlo simulation.

Key outcomes:
- Flat implied volatility appears as a model artifact
- Greeks measure local sensitivity, not global safety
- Tail risk is structurally underestimated

**Insight:**  
Black–Scholes does not describe markets — it describes a world where markets behave nicely.

---

## Module 4 — Beyond Black-Scholes: Model Risk & Hedging Failure

**Central idea:**  
Once volatility becomes stochastic, hedging certainty collapses.

This module explores stochastic volatility dynamics (Heston-style intuition), leverage effects, fat tails, and delta-hedging under model mismatch.

Key outcomes:
- Volatility and price become negatively correlated
- Extreme outcomes become structurally likely
- Delta-hedging accumulates persistent error
- Hedging error distributions develop heavy tails

**Insight:**  
Hedging fails not because traders are careless,  
but because the model assumes a world that does not exist.

---

## Final Conclusion

This project demonstrates a fundamental truth of quantitative finance:

> Models do not fail because they are wrong.  
> They fail because they are **believed too strongly**.

Risk is not volatility.
Risk is **model confidence**.

Understanding this distinction is the real edge.

---

## Technologies & Concepts Used

- Python
- NumPy
- Matplotlib
- Plotly
- Monte Carlo simulation
- Markov chains
- Black–Scholes framework
- Greeks
- Stochastic volatility
- Model risk analysis

---

## Disclaimer

This project is for educational and conceptual purposes only.
It is not financial advice and should not be used for live trading.

---


