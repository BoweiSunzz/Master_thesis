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
├── logs/
│   └── Final training results, performance comparisons, and analysis outputs
│
├── main.py    # Main script for integrating training  
│
├── evaluation.py   # Evaluation procedures and performance analysis
│
├── overall.ipynb    #  Overall comparison and visualization across different RL objective functions
 
    
```

## Running the Project

To train and evaluate the models, run: 

main.py

The training process generates multiple models for each objective function. The corresponding training results and model outputs are stored in the logs/ directory.

## Results

The evaluation.py is used to evaluate models generated under the same objective function and identify the best-performing model based on the defined performance metrics.

The final selected models from different objective functions are further compared using overall.ipynb. The analysis includes portfolio P&L statistics, risk metrics, hedging costs, and performance comparisons between RL-based and classical Greek-based hedging strategies.
