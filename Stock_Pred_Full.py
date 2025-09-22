import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
import hashlib
import time
from datetime import datetime, timedelta

# -------------------------------
# 1. FUNCTIONS
# -------------------------------

def get_stock_data(stock_symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for {stock_symbol}")
    return data[['Close']].dropna()

def create_dataset(dataset, time_step=60):
    """Create time windows for supervised learning."""
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def create_lstm_model(structure, input_shape):
    """Build different LSTM model structures."""
    model = Sequential()
    if structure == 'basic':
        model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    elif structure == 'stacked':
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
    elif structure == 'dropout':
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_data_hash(data_array):
    """"Generate SHA-256 hash of prediction data."""
    return hashlib.sha256(json.dumps(data_array.tolist()).encode()).hexdigest()

def store_on_blockchain(data, prev_hash="0"*64):
    """Simulate blockchain storage."""
    block_data = {"timestamp": datetime.now().isoformat(), "data_hash": create_data_hash(data), "prev_hash": prev_hash}
    block_hash = hashlib.sha256(json.dumps(block_data, sort_keys=True).encode()).hexdigest()
    print(f"Block stored: {block_hash[:10]}...")
    return block_hash

# -------------------------------
# 2. MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    # Παράμετροι
    stock_symbol = "AAPL"
    start_date = "2018-01-01"
    end_date = "2023-12-31"
    time_step = 60

    # Step 1: Load Data
    print(f"Fetching data for {stock_symbol}...")
    data = get_stock_data(stock_symbol, start_date, end_date)
    close_prices = data['Close'].values.reshape(-1, 1)
    dates = data.index
    print(f"Retrieved {len(data)} days of data.")

    # Step 2: Preprocessing
    print("Preprocessing data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    X, y = create_dataset(scaled_data, time_step)

    # Split 70% train, 15% val, 15% test
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    test_size = len(X) - train_size - val_size

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_lstm = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    X_train_rf = X_train
    X_test_rf = X_test

    # Step 3: Train LSTM Models
    print("\nTraining different LSTM structures...")
    lstm_structures = ['basic', 'stacked', 'dropout']
    lstm_results = {}

    for struct in lstm_structures:
        print(f"Training {struct.upper()} LSTM...")
        model = create_lstm_model(structure=struct, input_shape=(time_step, 1))
        model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(X_val_lstm, y_val), verbose=0)
        
        y_pred_scaled = model.predict(X_test_lstm)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mse = mean_squared_error(y_test_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, y_pred)
        
        lstm_results[struct] = {'model': model, 'mse': mse, 'rmse': rmse, 'r2': r2, 'pred': y_pred}

    # Step 4: Train Random Forest Models
    print("\nΔΟΚΙΜΗ ΔΙΑΦΟΡΕΤΙΚΩΝ ΔΟΜΩΝ RANDOM FOREST...")
    rf_structures = {
        'basic': {'n_estimators': 100, 'max_depth': None},
        'tuned_trees': {'n_estimators': 200, 'max_depth': None},
        'tuned_depth': {'n_estimators': 100, 'max_depth': 10},
    }
    rf_results = {}

    for name, params in rf_structures.items():
        print(f"Training  {name} RF...")
        rf_model = RandomForestRegressor(**params, random_state=42)
        rf_model.fit(X_train_rf, y_train)
        
        y_pred = rf_model.predict(X_test_rf)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mse = mean_squared_error(y_test_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, y_pred)
        
        rf_results[name] = {'model': rf_model, 'mse': mse, 'rmse': rmse, 'r2': r2, 'pred': y_pred}

    # Step 5: Grid Search + Cross Validation
    print("\nRunning Grid Search & Cross-Validation...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                               param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
    grid_search.fit(X_train_rf, y_train)
    best_rf_params = grid_search.best_params_
    print(f"Best Parameters: {best_rf_params}")

    # Cross-Validation
    cv_scores = cross_val_score(grid_search.best_estimator_, X_train_rf, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_cv = np.sqrt(-cv_scores)
    print(f"RF CV RMSE: {rmse_cv}, Avg: {np.mean(rmse_cv):.3f}")

    # Best RF model predictions
    y_pred_best_rf = grid_search.predict(X_test_rf)
    y_pred_best_rf = scaler.inverse_transform(y_pred_best_rf.reshape(-1, 1))
    mse_best = mean_squared_error(y_test_actual, y_pred_best_rf)
    rmse_best = np.sqrt(mse_best)
    r2_best = r2_score(y_test_actual, y_pred_best_rf)
    rf_results['best_grid'] = {'model': grid_search.best_estimator_, 'mse': mse_best, 'rmse': rmse_best, 'r2': r2_best, 'pred': y_pred_best_rf}

    # Step 6: Blockchain Storage Simulation
    print("Storing predictions on blockchain...")
    prev_hash = "0"*64
    for res in lstm_results.values():
        prev_hash = store_on_blockchain(res['pred'], prev_hash)
    for res in rf_results.values():
        prev_hash = store_on_blockchain(res['pred'], prev_hash)

    # Step 7: Generate Plots
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(18, 20))

    # Graph 1: Train/Val/Test Split
    ax1 = plt.subplot(4, 2, 1)
    full_data = scaler.inverse_transform(scaled_data)
    x_full = np.arange(len(full_data))
    ax1.plot(x_full, full_data, label='All Data', color='gray', alpha=0.7)
    ax1.set_xlim(0, len(full_data)-1)
    train_end = train_size + time_step
    val_end = train_end + val_size
    test_end = val_end + test_size
    ax1.axvspan(0, train_end, color='green', alpha=0.3, label='Εκπαίδευση (70%)')
    ax1.axvspan(train_end, val_end, color='blue', alpha=0.3, label='Επικύρωση (15%)')
    ax1.axvspan(val_end, test_end, color='red', alpha=0.3, label='Έλεγχος (15%)')
    ax1.set_title('Train/Validation/Test Split', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Graph 2: LSTM Structures RMSE
    ax2 = plt.subplot(4, 2, 2)
    labels = list(lstm_results.keys())
    rmse_vals = [lstm_results[k]['rmse'] for k in labels]
    bars = ax2.bar(labels, rmse_vals, color=['skyblue', 'lightcoral', 'gold'])
    ax2.set_title('LSTM Structures Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE')
    ax2.set_xlabel('Δομή LSTM')
    for bar, value in zip(bars, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.3f}', ha='center', va='bottom')
    ax2.grid(True, axis='y', alpha=0.3)

    # Graph 3: Random Forest Structures RMSE
    ax3 = plt.subplot(4, 2, 3)
    labels_rf = list(rf_results.keys())
    rmse_vals_rf = [rf_results[k]['rmse'] for k in labels_rf]
    bars = ax3.bar(labels_rf, rmse_vals_rf, color=['lightblue', 'lightcoral', 'lightgreen', 'gold'])
    ax3.set_title('Random Forest Structures Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RMSE')
    ax3.set_xlabel('Δομή Random Forest')
    for bar, value in zip(bars, rmse_vals_rf):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.3f}', ha='center', va='bottom')
    ax3.grid(True, axis='y', alpha=0.3)

    # Graph 4: Predictions vs Actual (LSTM)
    ax4 = plt.subplot(4, 2, 4)
    best_lstm = min(lstm_results.items(), key=lambda x: x[1]['rmse'])[0]
    test_dates = dates[-len(y_test):]
    ax4.plot(test_dates, y_test_actual, label='Actual Values', color='blue', linewidth=2)
    ax4.plot(test_dates, lstm_results[best_lstm]['pred'], label=f'LSTM Prediction ({best_lstm})', color='red', alpha=0.8, linewidth=2)
    ax4.set_title('Predictions vs Actual (LSTM)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Stock Price')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # Graph 5: Predictions vs Actual (Random Forest)
    ax5 = plt.subplot(4, 2, 5)
    best_rf = min(rf_results.items(), key=lambda x: x[1]['rmse'])[0]
    ax5.plot(test_dates, y_test_actual, label='Actual Values', color='blue', linewidth=2)
    ax5.plot(test_dates, rf_results[best_rf]['pred'], label=f'RF Prediction ({best_rf})', color='green', alpha=0.8, linewidth=2)
    ax5.set_title('Predictions vs Actual (Random Forest)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Stock Price')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # Graph 6: RMSE Comparison LSTM vs RF
    ax6 = plt.subplot(4, 2, 6)
    models = ['LSTM', 'Random Forest']
    rmse_final = [lstm_results[best_lstm]['rmse'], rf_results[best_rf]['rmse']]
    bars = ax6.bar(models, rmse_final, color=['red', 'green'], alpha=0.7)
    ax6.set_title('RMSE Comparison (LSTM vs RF)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('RMSE')
    ax6.set_ylim(0, max(rmse_final)*1.2)
    for bar, value in zip(bars, rmse_final):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_final)*0.01,
                 f'{value:.3f}', ha='center', va='bottom')
    ax6.grid(True, axis='y', alpha=0.3)

    
    # Graph 7: Visualization with Dates
    ax8 = plt.subplot(4, 2, 8)
    n_show = 150
    dates_show = test_dates[-n_show:]
    y_test_show = y_test_actual[-n_show:].flatten()
    y_pred_lstm_show = lstm_results[best_lstm]['pred'][-n_show:].flatten()
    y_pred_rf_show = rf_results[best_rf]['pred'][-n_show:].flatten()
    ax8.plot(dates_show, y_test_show, label='Actual Values', color='blue', linewidth=2)
    ax8.plot(dates_show, y_pred_lstm_show, label='LSTM Prediction', color='red', alpha=0.8, linestyle='--')
    ax8.plot(dates_show, y_pred_rf_show, label='RF Prediction', color='green', alpha=0.8, linestyle='--')
    ax8.set_title('Predictions with Dates', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Stock Price')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout(pad=3.0)
    plt.savefig('full_results_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Step 8: Next-Day Prediction
    print("\n" + "="*50)
    print("NEXT DAY PREDICTION")
    print("="*50)

    # LSTM next-day prediction
    last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
    best_lstm_model = lstm_results[best_lstm]['model']
    lstm_pred_scaled = best_lstm_model.predict(last_sequence, verbose=0)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)[0][0]

    # Random Forest next-day prediction
    rf_input = scaled_data[-time_step:].reshape(1, -1)
    rf_pred_scaled = rf_results[best_rf]['model'].predict(rf_input)
    rf_pred = scaler.inverse_transform(rf_pred_scaled.reshape(-1, 1))[0][0]

    next_date = dates[-1] + timedelta(days=1)
    print(f"Last Date: {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Last Price: {close_prices[-1][0]:.2f}")
    print(f"LSTM Prediction for {next_date.strftime('%Y-%m-%d')}: {lstm_pred:.2f}")
    print(f"RF Prediction for {next_date.strftime('%Y-%m-%d')}: {rf_pred:.2f}")

    # Save prediction to blockchain
    prediction_data = np.array([[lstm_pred], [rf_pred]])
    store_on_blockchain(prediction_data, prev_hash)