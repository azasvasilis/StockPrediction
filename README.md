# Stock Prediction with LSTM and Random Forest

This project implements a stock price prediction system using **LSTM Neural Networks** and **Random Forest Regressors**.  
It also includes **data preprocessing**, **model evaluation**, and a simple **blockchain simulation** to store predictions.

## Features
- Download historical stock data from **Yahoo Finance**
- Data preprocessing with scaling and time-series window creation
- Training and comparison of:
  - Multiple LSTM architectures (basic, stacked, dropout)
  - Random Forest with different parameters + Grid Search
- Performance evaluation with RMSE and R²
- Visualization of results with matplotlib
- Simple blockchain simulation for immutable storage of predictions

## Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
Usage

Run the script:

python stock_prediction.py


This will:

Download stock data

Train the models

Save plots in full_results_analysis.png

Print next-day prediction from both models

Output

Comparison plots (LSTM vs Random Forest)

Blockchain storage simulation of predictions

Next-day stock price prediction

Example
Last Date: 2023-12-29
Last Close Price: 192.53
LSTM Prediction for 2024-01-02: 194.12
RF Prediction for 2024-01-02: 193.47


✍️ Author: Vasilis Azas
📌 GitHub: vasilisazas