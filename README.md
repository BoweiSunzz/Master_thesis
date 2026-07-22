# Master Thesis: Objective Function Design in Reinforcement Learning for Deep Hedging

This repository contains the source code and experimental results for my Master's thesis. The project investigates how different RL objective functions influence hedging performance, training stability, and robustness in a dynamic option hedging framework.

## Project Overview

The thesis implements a QR-D4PG agent for dynamic hedging of option portfolios. The RL-based hedging strategies are compared with classical Greek-based hedging methods, including Delta, Delta-Gamma, and Delta-Vega hedging.

The environment simulates stochastic asset prices and volatility dynamics, and client option arrivals are modeled using a Poisson process. The final evaluation is conducted on additional simulated episodes, and the results are stored in the `logs/` directory.

## Repository Structure
```text
Master_thesis/
│
├── agent/
│   └── RL agent construction, training, and Greek-based hedging agents
│
├── checkpoints/
│   └── Saved model checkpoints during training
│
├── environment/
│   └── Simulation environment and utility functions for the thesis framework
│
├── evaluation/
│   └── Evaluation procedures and performance analysis
│
├── logs/
│   └── Final training results, performance comparisons, and analysis outputs
│
├── main.py
    └── Main script for integrating training
